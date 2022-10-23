import argparse
import os
import pickle
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse
import torch
from harmony import harmonize

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

sys.path.append("../resources")
from resources.data import ModalityMatchingDataset
from resources.models import Modality_CLIP, Encoder, defaults_GEX2ADT, defaults_GEX2ATAC
from resources.OTmatching import get_OT_bipartite_matching_adjacency_matrix

# Parse args
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='TASK')
parser.add_argument('--epsilon', '-H', type=float, default=0.01,
                    description='entropy regularization strength for the OT matching')
parser_GEX2ADT = subparsers.add_parser('GEX2ADT', help='train GEX2ADT model')
for key, value in defaults_GEX2ADT.items():
    parser_GEX2ADT.add_argument("--" + key, default=value, type=type(value))
parser_GEX2ATAC = subparsers.add_parser('GEX2ATAC', help='train GEX2ATAC model')
for key, value in defaults_GEX2ATAC.items():
    parser_GEX2ATAC.add_argument("--" + key, default=value, type=type(value))
args, unknown_args = parser.parse_known_args()

# Define file paths
if args.TASK == 'GEX2ADT':
    dataset_path = "../datasets/phase2-private-data/match_modality/openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_"
    pretrain_path = "../pretrainGEX2ADT"
    is_multiome = False
elif args.TASK == 'GEX2ATAC':
    dataset_path = "../datasets/phase2-private-data/match_modality/openproblems_bmmc_multiome_phase2_rna/openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_"
    pretrain_path = "../pretrainGEX2ATAC"
    is_multiome = True
else:
    raise ValueError('Unknown task: ' + args.TASK)

par = {
    "input_train_mod1": f"{dataset_path}train_mod1.h5ad",
    "input_train_mod2": f"{dataset_path}train_mod2.h5ad",
    "input_test_mod1": f"{dataset_path}test_mod1.h5ad",
    "input_test_mod2": f"{dataset_path}test_mod2.h5ad",
    "input_pretrain": pretrain_path,
    "output": args.TASK + "H" + str(args.epsilon) + ".h5ad",
}

# Load data
input_train_mod1 = ad.read_h5ad(par["input_train_mod1"])
input_train_mod2 = ad.read_h5ad(par["input_train_mod2"])
input_test_mod1 = ad.read_h5ad(par["input_test_mod1"])
input_test_mod2 = ad.read_h5ad(par["input_test_mod2"])

N_cells = input_test_mod1.shape[0]
assert input_test_mod2.shape[0] == N_cells

# Load and apply LSI transformation, transductive setting
with open(par["input_pretrain"] + "/lsi_GEX_transformer.pickle", "rb") as f:
    lsi_transformer_gex = pickle.load(f)
if is_multiome:
    with open(par["input_pretrain"] + "/lsi_ATAC_transformer.pickle", "rb") as f:
        lsi_transformer_atac = pickle.load(f)
    gex_train = lsi_transformer_gex.transform(input_train_mod1)
    gex_test = lsi_transformer_gex.transform(input_test_mod1)
    atac_train = lsi_transformer_atac.transform(input_train_mod2)
    atac_test = lsi_transformer_atac.transform(input_test_mod2)
else:
    gex_train = lsi_transformer_gex.transform(input_train_mod1)
    gex_test = lsi_transformer_gex.transform(input_test_mod1)
    adt_train = input_train_mod2.to_df()
    adt_test = input_test_mod2.to_df()

# Apply Harmony transformation, transductive setting
gex_train['batch'] = input_train_mod1.obs.batch
gex_test['batch'] = input_test_mod1.obs.batch
if is_multiome:
    atac_train['batch'] = input_train_mod2.obs.batch
    atac_test['batch'] = input_test_mod2.obs.batch
else:
    adt_train['batch'] = input_train_mod2.obs.batch
    adt_test['batch'] = input_test_mod2.obs.batch

def harmony_transductive(train, test, use_gpu=True):
    all = pd.concat([train, test])
    all_batches = all.pop('batch')
    all_batches.columns = ['batch']
    all_batches = all_batches.to_frame()
    all_harmony = harmonize(all.to_numpy(), all_batches, batch_key='batch', use_gpu=use_gpu, verbose=True)
    train_harmony = all_harmony[:len(train)]
    test_harmony = all_harmony[len(train):]
    return train_harmony, test_harmony

gex_train_h, gex_test_h = harmony_transductive(gex_train, gex_test)
if is_multiome:
    # mod2 is ATAC
    mod2_train_h, mod2_test_h = harmony_transductive(atac_train, atac_test)
else:
    # mod2 is ADT
    mod2_train_h, mod2_test_h = harmony_transductive(adt_train, adt_test)

# Load pretrained models and ensemble predictions
sim_matrix = np.zeros((N_cells, N_cells))
for fold in range(0, 9):
    weight_file = par["input_pretrain"] + "/" + str(fold) + "/model.best.pth"
    if os.path.exists(weight_file):
        print("Loading weights from " + weight_file)
        weight = torch.load(weight_file, map_location="cpu")

        # Define modality encoders
        if is_multiome:
            model = Modality_CLIP(
                Encoder=Encoder,
                layers_dims=(
                    [args.LAYERS_DIM_ATAC0, args.LAYERS_DIM_ATAC1],
                    [args.LAYERS_DIM_GEX0, args.LAYERS_DIM_GEX1],
                ),
                dropout_rates=(
                    [args.DROPOUT_RATES_ATAC0, args.DROPOUT_RATES_ATAC1],
                    [args.DROPOUT_RATES_GEX0, args.DROPOUT_RATES_GEX1],
                ),
                dim_mod1=args.N_LSI_COMPONENTS_ATAC,
                dim_mod2=args.N_LSI_COMPONENTS_GEX,
                output_dim=args.EMBEDDING_DIM,
                T=args.LOG_T,
                noise_amount=args.SFA_NOISE,
            ).to(device)
        else:
            model = Modality_CLIP(
                Encoder=Encoder,
                layers_dims=(
                    [args.LAYERS_DIM_ADT],
                    [args.LAYERS_DIM_GEX0, args.LAYERS_DIM_GEX1],
                ),
                dropout_rates=(
                    [args.DROPOUT_RATES_ADT],
                    [args.DROPOUT_RATES_GEX0, args.DROPOUT_RATES_GEX1],
                ),
                dim_mod1=args.N_LSI_COMPONENTS_ADT,
                dim_mod2=args.N_LSI_COMPONENTS_GEX,
                output_dim=args.EMBEDDING_DIM,
                T=args.LOG_T,
                noise_amount=args.SFA_NOISE,
            ).to(device)

        # Load pretrained weights
        model.load_state_dict(weight)

        # Load torch datasets
        dataset_test = ModalityMatchingDataset(gex_test_h, mod2_test_h)
        data_test = torch.utils.data.DataLoader(dataset_test, 32, shuffle=False)

        # Predict on test set
        all_emb_mod1 = []
        all_emb_mod2 = []
        indexes = []
        model.eval()
        for x1, x2 in data_test:
            x1, x2 = x1.float(), x2.float()
            # The model applies the GEX encoder to the second argument, here x1
            logits, features_mod2, features_mod1 = model(
                x2.to("cuda"), x1.to("cuda")
            )

            all_emb_mod1.append(features_mod1.detach().cpu())
            all_emb_mod2.append(features_mod2.detach().cpu())

        all_emb_mod1 = torch.cat(all_emb_mod1)
        all_emb_mod2 = torch.cat(all_emb_mod2)

        # Calculate the cosine similarity matrix and add it to the ensemble
        sim_matrix += (all_emb_mod1 @ all_emb_mod2.T).detach().cpu().numpy()

# Split matching by batch label
mod1_splits = set(input_test_mod1.obs["batch"])
mod2_splits = set(input_test_mod2.obs["batch"])
splits = mod1_splits | mod2_splits
matching_matrices, mod1_obs_names, mod2_obs_names = [], [], []
mod1_obs_index = input_test_mod1.obs.index
mod2_obs_index = input_test_mod2.obs.index

for split in splits:
    print("matching split", split)
    mod1_split = input_test_mod1[input_test_mod1.obs["batch"] == split]
    mod2_split = input_test_mod2[input_test_mod2.obs["batch"] == split]
    mod1_obs_names.append(mod1_split.obs_names)
    mod2_obs_names.append(mod2_split.obs_names)
    mod1_indexes = mod1_obs_index.get_indexer(mod1_split.obs_names)
    mod2_indexes = mod2_obs_index.get_indexer(mod2_split.obs_names)
    sim_matrix_split = sim_matrix[np.ix_(mod1_indexes, mod2_indexes)]

    # Compute OT matching
    matching_matrices.append(
        get_OT_bipartite_matching_adjacency_matrix(sim_matrix_split, epsilon=args.epsilon)
    )

# Assemble the matching matrices and reorder according to the original order of cell profiles
matching_matrix = scipy.sparse.block_diag(matching_matrices, format="csc")
mod1_obs_names = pd.Index(np.concatenate(mod1_obs_names))
mod2_obs_names = pd.Index(np.concatenate(mod2_obs_names))
matching_matrix = matching_matrix[mod1_obs_names.get_indexer(mod1_obs_index), :][
                  :, mod2_obs_names.get_indexer(mod2_obs_index)]

# Save the matching matrix
out = ad.AnnData(
        X=matching_matrix,
        uns={
            "dataset_id": input_test_mod1.uns["dataset_id"],
            "method_id": "OT",
        },
    )
out.write_h5ad(par["output"], compression="gzip")
