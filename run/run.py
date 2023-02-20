import argparse
import os
import pickle
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse
import torch
from distutils.util import strtobool

from catalyst.utils import set_global_seed

sys.path.append(".")
from resources.data import ModalityMatchingDataset
from resources.models import Modality_CLIP, Encoder
from resources.postprocessing import OT_matching, MWB_matching
from resources.hyperparameters import *
from resources.preprocessing import harmony
from evaluate import evaluate

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Define argument parsers
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='TASK')

# Common args
for key, value in defaults_common.items():
    parser.add_argument("--" + key, default=value,
                        type=(lambda x: bool(strtobool(x))) if type(value) == bool else type(value))

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

# Set global random seed
set_global_seed(args.SEED)

# Define file paths
if args.TASK == 'GEX2ADT':
    dataset_path = os.path.join(args.DATASETS_PATH,
                                "openproblems_bmmc_cite_phase2_rna/"
                                "openproblems_bmmc_cite_phase2_rna.censor_dataset.output_")
    pretrain_path = os.path.join(args.PRETRAIN_PATH, "GEX2ADT")
    is_multiome = False
elif args.TASK == 'GEX2ATAC':
    dataset_path = os.path.join(args.DATASETS_PATH,
                                "openproblems_bmmc_multiome_phase2_rna/"
                                "openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_")
    pretrain_path = os.path.join(args.PRETRAIN_PATH, "GEX2ATAC")
    is_multiome = True
else:
    raise ValueError('Unknown task: ' + args.TASK)
if args.CUSTOM_DATASET_PATH != '':
    dataset_path = args.CUSTOM_DATASET_PATH
    assert args.TRANSDUCTIVE == False
    assert args.HARMONY == False

par = {
    "input_train_mod1": f"{dataset_path}train_mod1.h5ad",
    "input_train_mod2": f"{dataset_path}train_mod2.h5ad",
    "input_test_mod1": f"{dataset_path}test_mod1.h5ad",
    "input_test_mod2": f"{dataset_path}test_mod2.h5ad",
    "input_pretrain": pretrain_path,
    "output": os.path.join(args.PRETRAIN_PATH, args.OUT_NAME + args.TASK)
}

# Overwrite configurations for ablation study
if args.HYPERPARAMS == False:
    if is_multiome:
        for hyperparam, baseline_value in baseline_GEX2ATAC.items():
            setattr(args, hyperparam, baseline_value)
    else:
        for hyperparam, baseline_value in baseline_GEX2ADT.items():
            setattr(args, hyperparam, baseline_value)
print("args:", args, "unknown_args:", unknown_args)

# Load data
if args.TRANSDUCTIVE:
    input_train_mod1 = ad.read_h5ad(par["input_train_mod1"])
    input_train_mod2 = ad.read_h5ad(par["input_train_mod2"])
input_test_mod1 = ad.read_h5ad(par["input_test_mod1"])
input_test_mod2 = ad.read_h5ad(par["input_test_mod2"])

N_cells = input_test_mod1.shape[0]
print("mod1 cells:", input_test_mod1.shape[0], "mod2 cells:", input_test_mod2.shape[0])
assert input_test_mod2.shape[0] == N_cells

# Load and apply LSI transformation
with open(par["input_pretrain"] + "/lsi_GEX_transformer.pickle", "rb") as f:
    lsi_transformer_gex = pickle.load(f)
if is_multiome:
    with open(par["input_pretrain"] + "/lsi_ATAC_transformer.pickle", "rb") as f:
        lsi_transformer_atac = pickle.load(f)
    if args.TRANSDUCTIVE:
        gex_train = lsi_transformer_gex.transform(input_train_mod1)
        mod2_train = lsi_transformer_atac.transform(input_train_mod2)
    gex_test = lsi_transformer_gex.transform(input_test_mod1)
    mod2_test = lsi_transformer_atac.transform(input_test_mod2)
else:
    if args.TRANSDUCTIVE:
        gex_train = lsi_transformer_gex.transform(input_train_mod1)
        mod2_train = input_train_mod2.to_df()
    gex_test = lsi_transformer_gex.transform(input_test_mod1)
    mod2_test = input_test_mod2.to_df()

if args.HARMONY:
    # Apply Harmony batch effect correction
    gex_test['batch'] = input_test_mod1.obs.batch
    mod2_test['batch'] = input_test_mod2.obs.batch

    if args.TRANSDUCTIVE:
        # Transductive setting
        gex_train['batch'] = input_train_mod1.obs.batch
        gex_train, gex_test, = harmony([gex_train, gex_test])
        mod2_train['batch'] = input_train_mod2.obs.batch
        mod2_train, mod2_test, = harmony([mod2_train, mod2_test])
    else:
        gex_test, = harmony([gex_test])
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
        dataset_test = ModalityMatchingDataset(pd.DataFrame(gex_test), pd.DataFrame(mod2_test))
        data_test = torch.utils.data.DataLoader(dataset_test, 32, shuffle=False)

        # Predict on test set
        all_emb_mod1 = []
        all_emb_mod2 = []
        indexes = []
        model.eval()
        for batch in data_test:
            x1 = batch["features_first"].float()
            x2 = batch["features_second"].float()
            # The model applies the GEX encoder to the second argument, here x1
            logits, features_mod2, features_mod1 = model(
                x2.to(device), x1.to(device)
            )

            all_emb_mod1.append(features_mod1.detach().cpu())
            all_emb_mod2.append(features_mod2.detach().cpu())

        all_emb_mod1 = torch.cat(all_emb_mod1)
        all_emb_mod2 = torch.cat(all_emb_mod2)

        # Save the embeddings concatenated according to the true order and predicted order
        if args.SAVE_EMBEDDINGS:
            # Assumes that the two modalities have the cells in the same order
            all_emb_mod12_true = torch.cat((all_emb_mod1, all_emb_mod2), dim=1)
            file = par["output"] + "emb_mod12_fold" + str(fold) + "_truematch.pt"
            torch.save(all_emb_mod12_true, file)
            print("Prediction saved to", file)
            predicted_match = ad.read_h5ad(par["output"] + ".h5ad")
            Xsol = torch.tensor(predicted_match.X)
            all_emb_mod12_pred = torch.cat((all_emb_mod1, all_emb_mod2[Xsol.argmax(1)]), dim=1)
            file = par["output"] + "emb_mod12_fold" + str(fold) + "_predmatch.pt"
            torch.save(all_emb_mod12_pred, file)
            print("Prediction saved to", file)

        # Calculate the cosine similarity matrix and add it to the ensemble
        sim_matrix += (all_emb_mod1 @ all_emb_mod2.T).detach().cpu().numpy()

# save the full similarity matrix
# np.save("similarity_matrix.npy", sim_matrix)

if args.BATCH_LABEL_MATCHING:
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
        # save the split similarity matrix
        # np.save("test_similarity_matrix"+str(split)+".npy", sim_matrix_split)

        if args.OT_MATCHING:
            # Compute OT matching
            matching_matrices.append(
                OT_matching(sim_matrix_split, entropy_reg=args.OT_ENTROPY)
            )
        else:
            # Max-weight bipartite matching
            matching_matrices.append(
                MWB_matching(sim_matrix_split)
            )

    # Assemble the matching matrices and reorder according to the original order of cell profiles
    matching_matrix = scipy.sparse.block_diag(matching_matrices, format="csc")
    mod1_obs_names = pd.Index(np.concatenate(mod1_obs_names))
    mod2_obs_names = pd.Index(np.concatenate(mod2_obs_names))
    matching_matrix = matching_matrix[mod1_obs_names.get_indexer(mod1_obs_index), :][
                      :, mod2_obs_names.get_indexer(mod2_obs_index)]
else:
    if args.OT_MATCHING:
        # Compute OT matching
        matching_matrix = OT_matching(sim_matrix, entropy_reg=args.OT_ENTROPY)
    else:
        # Max-weight bipartite matching
        matching_matrix = MWB_matching(sim_matrix)
    matching_matrix = pd.DataFrame(matching_matrix)

out = ad.AnnData(
    X=matching_matrix,
    uns={
        # "dataset_id": input_test_mod1.uns["dataset_id"],
        "method_id": "MatchCLOT",
    },
)

# Save the matching matrix
out.write_h5ad(par["output"] + ".h5ad", compression="gzip")
print("Prediction saved to", par["output"] + ".h5ad")

# Load the solution for evaluation
if is_multiome:
    sol_path = os.path.join(args.DATASETS_PATH, "openproblems_bmmc_multiome_phase2_rna"
                                                "/openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_")
else:
    sol_path = os.path.join(args.DATASETS_PATH, "openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna"
                                                ".censor_dataset.output_")
sol = ad.read_h5ad(sol_path + "test_sol.h5ad")

# Score the prediction and save the results
scores_path = os.path.join("scores", args.OUT_NAME + args.TASK + ".txt")
evaluate(out, sol, scores_path)


