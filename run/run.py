import argparse
import os
import pickle
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse
import torch

from resources.preprocessing import harmony

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

sys.path.append("../resources")
from resources.data import ModalityMatchingDataset
from resources.models import Modality_CLIP, Encoder
from resources.OTmatching import get_OT_bipartite_matching_adjacency_matrix
from resources.hyperparameters import defaults_common, defaults_GEX2ADT, defaults_GEX2ATAC, baseline_GEX2ADT, \
    baseline_GEX2ATAC

# Define argument parsers
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='TASK')

# Common args
for key, value in defaults_common.items():
    parser.add_argument("--" + key, default=value, type=type(value))

parser.add_argument('--epsilon', '-H', type=float, default=0.01,
                    description='entropy regularization strength for the OT matching')

# GEX2ADT args
parser_GEX2ADT = subparsers.add_parser('GEX2ADT', help='train GEX2ADT model')
for key, value in defaults_GEX2ADT.items():
    parser_GEX2ADT.add_argument("--" + key, default=value, type=type(value))

# GEX2ATAC args
parser_GEX2ATAC = subparsers.add_parser('GEX2ATAC', help='train GEX2ATAC model')
for key, value in defaults_GEX2ATAC.items():
    parser_GEX2ATAC.add_argument("--" + key, default=value, type=type(value))

# Parse args
args, unknown_args = parser.parse_known_args()

# Define file paths
if args.TASK == 'GEX2ADT':
    dataset_path = "../datasets/phase2-private-data/match_modality/openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_"
    pretrain_path = args.PRETRAIN_PATH
    is_multiome = False
elif args.TASK == 'GEX2ATAC':
    dataset_path = "../datasets/phase2-private-data/match_modality/openproblems_bmmc_multiome_phase2_rna/openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_"
    pretrain_path = args.PRETRAIN_PATH
    is_multiome = True
else:
    raise ValueError('Unknown task: ' + args.TASK)

par = {
    "input_train_mod1": f"{dataset_path}train_mod1.h5ad",
    "input_train_mod2": f"{dataset_path}train_mod2.h5ad",
    "input_test_mod1": f"{dataset_path}test_mod1.h5ad",
    "input_test_mod2": f"{dataset_path}test_mod2.h5ad",
    "input_pretrain": pretrain_path,
    "output": args.TASK + ".h5ad",
}

# Overwrite configurations for ablation study
if args.HYPERPARAMS == False:
    if is_multiome:
        for hyperparam, baseline_value in baseline_GEX2ATAC.items():
            args.hyperparam = baseline_value
    else:
        for hyperparam, baseline_value in baseline_GEX2ADT.items():
            args.hyperparam = baseline_value

# Load data
input_train_mod1 = ad.read_h5ad(par["input_train_mod1"])
input_train_mod2 = ad.read_h5ad(par["input_train_mod2"])
input_test_mod1 = ad.read_h5ad(par["input_test_mod1"])
input_test_mod2 = ad.read_h5ad(par["input_test_mod2"])

N_cells = input_test_mod1.shape[0]
assert input_test_mod2.shape[0] == N_cells

# Load and apply LSI transformation
with open(par["input_pretrain"] + "/lsi_GEX_transformer.pickle", "rb") as f:
    lsi_transformer_gex = pickle.load(f)
if is_multiome:
    with open(par["input_pretrain"] + "/lsi_ATAC_transformer.pickle", "rb") as f:
        lsi_transformer_atac = pickle.load(f)
    gex_train = lsi_transformer_gex.transform(input_train_mod1)
    gex_test = lsi_transformer_gex.transform(input_test_mod1)
    mod2_train = lsi_transformer_atac.transform(input_train_mod2)
    mod2_test = lsi_transformer_atac.transform(input_test_mod2)
else:
    gex_train = lsi_transformer_gex.transform(input_train_mod1)
    gex_test = lsi_transformer_gex.transform(input_test_mod1)
    mod2_train = input_train_mod2.to_df()
    mod2_test = input_test_mod2.to_df()

if args.HARMONY:
    # Apply Harmony batch effect correction
    gex_train['batch'] = input_train_mod1.obs.batch
    gex_test['batch'] = input_test_mod1.obs.batch
    mod2_train['batch'] = input_train_mod2.obs.batch
    mod2_test['batch'] = input_test_mod2.obs.batch

    if args.TRANSDUCTIVE:
        # Transductive setting
        gex_train, gex_test, = harmony([gex_train, gex_test])
        mod2_train, mod2_test, = harmony([mod2_train, mod2_test])
    else:
        gex_train, = harmony([gex_train])
        gex_test, = harmony([gex_test])
        mod2_train, = harmony([mod2_train])
        mod2_test, = harmony([mod2_test])

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
                    [args.LAYERS_DIM_ATAC],
                    [args.LAYERS_DIM_GEX0, args.LAYERS_DIM_GEX1],
                ),
                dropout_rates=(
                    [args.DROPOUT_RATES_ATAC],
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
                    [args.LAYERS_DIM_ADT0, args.LAYERS_DIM_ADT1],
                    [args.LAYERS_DIM_GEX0, args.LAYERS_DIM_GEX1],
                ),
                dropout_rates=(
                    [args.DROPOUT_RATES_ADT0, args.DROPOUT_RATES_ADT1],
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
        dataset_test = ModalityMatchingDataset(gex_test, mod2_test)
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
