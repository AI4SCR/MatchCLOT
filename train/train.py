import argparse
import os
import pickle
import sys

import anndata as ad
import pandas as pd
import torch
from catalyst import dl
from sklearn.model_selection import LeaveOneGroupOut
from distutils.util import strtobool

sys.path.append(".")
from resources.data import ModalityMatchingDataset
from resources.models import Modality_CLIP, Encoder
from resources.catalyst_tools import scRNARunner, CustomMetric
from resources.preprocessing import lsiTransformer, harmony
from resources.hyperparameters import *

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Define argument parsers
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='TASK')

# Common args
for key, value in defaults_common.items():
    parser.add_argument("--" + key, default=value, type=(lambda x: bool(strtobool(x))) if type(value) == bool else type(value))

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

# Date in format YYMMDDHHMMSS
date = ''.join([c if c.isnumeric() else '' for c in str(pd.Timestamp('today').to_pydatetime())][2:19])

# Define file paths
if args.TASK == 'GEX2ADT':
    dataset_path = os.path.join(args.DATASETS_PATH, "openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_")
    pretrain_path = os.path.join(args.PRETRAIN_PATH, "GEX2ADT")    # Path for saving the trained model
    is_multiome = False
elif args.TASK == 'GEX2ATAC':
    dataset_path = os.path.join(args.DATASETS_PATH, "openproblems_bmmc_multiome_phase2_rna/openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_")
    pretrain_path = os.path.join(args.PRETRAIN_PATH, "GEX2ATAC")
    is_multiome = True
else:
    raise ValueError('Unknown task: ' + args.TASK)

par = {
    "input_train_mod1": f"{dataset_path}train_mod1.h5ad",
    "input_train_mod2": f"{dataset_path}train_mod2.h5ad",
    "input_train_sol": f"{dataset_path}train_sol.h5ad",
    "input_test_mod1": f"{dataset_path}test_mod1.h5ad",
    "input_test_mod2": f"{dataset_path}test_mod2.h5ad",
    "output_pretrain": pretrain_path,
    "input_pretrain": pretrain_path,
}
os.makedirs(par["output_pretrain"], exist_ok=True)

# Overwrite configurations for ablation study
if args.HYPERPARAMS == False:
    if is_multiome:
        for hyperparam, baseline_value in baseline_GEX2ATAC.items():
            setattr(args, hyperparam, baseline_value)
    else:
        for hyperparam, baseline_value in baseline_GEX2ADT.items():
            setattr(args, hyperparam, baseline_value)
print("args:", args, "unknown_args:", unknown_args)

# Load train data
print("loading train data")
input_train_mod1 = ad.read_h5ad(par["input_train_mod1"])
print("input_train_mod1.shape", input_train_mod1.shape)
input_train_mod2 = ad.read_h5ad(par["input_train_mod2"])
print("input_train_mod2.shape", input_train_mod2.shape)
sol_train = ad.read_h5ad(par["input_train_sol"])  # ground truth matching

mod1 = input_train_mod1.var["feature_types"][0]
mod2 = input_train_mod2.var["feature_types"][0]
assert mod1 == "GEX"  # mod1 is always GEX, mod2 is either ADT or ATAC

# Apply the same ordering of mod2 profiles as mod1
input_train_mod2 = input_train_mod2[sol_train.to_df().values.argmax(1)]

# Load private test data, used for transductive LSI + Harmony preprocessing
print("loading private test data")
input_test_mod1 = ad.read_h5ad(par["input_test_mod1"])
print("input_test_mod1.shape", input_test_mod1.shape)
input_test_mod2 = ad.read_h5ad(par["input_test_mod2"])
print("input_test_mod2.shape", input_test_mod2.shape)

# Define train and validation split
fold_number = args.VALID_FOLD
print("fold_number:", fold_number)
trial_dump_folder = os.path.join(par["output_pretrain"], str(fold_number))
logo = LeaveOneGroupOut()
groups = sol_train.obs.batch
print("GROUPS:", groups)
logo.get_n_splits(input_train_mod2, groups=groups)
all_splits = list(logo.split(input_train_mod2, groups=groups))
train_indexes, test_indexes = all_splits[fold_number]
print("len train:", len(train_indexes), "len test:", len(test_indexes))

# Load or fit LSI preprocessing
path = par["output_pretrain"]

if os.path.exists(path + "/lsi_GEX_transformer.pickle") and args.TRANSDUCTIVE and not is_multiome:
    # Avoid re-computing LSI transformation when using cross-validation and transductive LSI
    print("loading lsi transformer from", path)
    # LSI is applied only on GEX and ATAC, not on ADT
    with open(path + "/lsi_GEX_transformer.pickle", "rb") as f:
        lsi_transformer_gex = pickle.load(f)
elif os.path.exists(path + "/lsi_GEX_transformer.pickle") and\
     os.path.exists(path + "/lsi_ATAC_transformer.pickle") and args.TRANSDUCTIVE and is_multiome:
    with open(path + "/lsi_GEX_transformer.pickle", "rb") as f:
        lsi_transformer_gex = pickle.load(f)
    with open(path + "/lsi_ATAC_transformer.pickle", "rb") as f:
        lsi_transformer_atac = pickle.load(f)
else:
    print("No lsi transformer found in", path, "creating new one")
    os.makedirs(path, exist_ok=True)

    # Fit GEX LSI
    lsi_transformer_gex = lsiTransformer(
        n_components=args.N_LSI_COMPONENTS_GEX, drop_first=True
    )
    if args.TRANSDUCTIVE:
        print("concatenating gex train and test")
        concatenated_gex = ad.concat([input_train_mod1, input_test_mod1], join="outer")
        print("done, concatenated_gex.shape", concatenated_gex.shape)
        lsi_transformer_gex.fit(concatenated_gex)
        # Save LSI transformation
        with open(path + "/lsi_GEX_transformer.pickle", "wb") as f:
            pickle.dump(lsi_transformer_gex, f)
        print("saved lsi pickle in ", path + "/lsi_GEX_transformer.pickle")
    else:
        lsi_transformer_gex.fit(input_train_mod1)
        with open(path + "/lsi_GEX_transformer.pickle", "wb") as f:
            pickle.dump(lsi_transformer_gex, f)
        print("saved lsi pickle in ", trial_dump_folder + "/lsi_GEX_transformer.pickle")

    # LSI is applied only on GEX and ATAC, not on ADT
    if is_multiome:
        # Fit ATAC LSI
        lsi_transformer_atac = lsiTransformer(
            n_components=args.N_LSI_COMPONENTS_ATAC, drop_first=True
        )
        if args.TRANSDUCTIVE:
            print("concatenating atac train and test")
            concatenated_atac = ad.concat([input_train_mod2, input_test_mod2], join="outer")
            print("done, concatenated_atac.shape", concatenated_atac.shape)
            lsi_transformer_atac.fit(concatenated_atac)
        else:
            lsi_transformer_atac.fit(input_train_mod2)

        # Save LSI transformation
        with open(path + "/lsi_ATAC_transformer.pickle", "wb") as f:
            pickle.dump(lsi_transformer_atac, f)
        print("saved lsi pickle in ", path + "/lsi_ATAC_transformer.pickle")

# Apply LSI preprocessing
gex_train = lsi_transformer_gex.transform(input_train_mod1[train_indexes])
gex_test = lsi_transformer_gex.transform(input_train_mod1[test_indexes])
gex_private = lsi_transformer_gex.transform(input_test_mod1)
if is_multiome:
    mod2_train = lsi_transformer_atac.transform(input_train_mod2[train_indexes])
    mod2_test = lsi_transformer_atac.transform(input_train_mod2[test_indexes])
    mod2_private = lsi_transformer_atac.transform(input_test_mod2)
else:
    mod2_train = input_train_mod2[train_indexes].to_df()
    mod2_test = input_train_mod2[test_indexes].to_df()
    mod2_private = input_test_mod2.to_df()

# Apply Harmony batch effect correction
if args.HARMONY:
    gex_train['batch'] = input_train_mod1.obs.batch[train_indexes]
    gex_test['batch'] = input_train_mod1.obs.batch[test_indexes]
    gex_private['batch'] = input_test_mod1.obs.batch
    mod2_train['batch'] = input_train_mod2.obs.batch[train_indexes]
    mod2_test['batch'] = input_train_mod2.obs.batch[test_indexes]
    mod2_private['batch'] = input_test_mod2.obs.batch

    if args.TRANSDUCTIVE:
        # Transductive setting
        gex_train, gex_test, gex_private = harmony([gex_train, gex_test, gex_private])
        mod2_train, mod2_test, mod2_private = harmony([mod2_train, mod2_test, mod2_private])
    else:
        gex_train, = harmony([gex_train])
        gex_test, = harmony([gex_test])
        mod2_train, = harmony([mod2_train])
        mod2_test, = harmony([mod2_test])

# Load torch dataloaders
dataset_train = ModalityMatchingDataset(pd.DataFrame(mod2_train), pd.DataFrame(gex_train))
dataset_test = ModalityMatchingDataset(pd.DataFrame(mod2_test), pd.DataFrame(gex_test))
dataloader_train = torch.utils.data.DataLoader(
    dataset_train, args.BATCH_SIZE, shuffle=True, num_workers=4
)
dataloader_test = torch.utils.data.DataLoader(
    dataset_test, 256, shuffle=False, num_workers=4
)
print("loaded dataloaders")

# Define modality encoders and trainer
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
    )
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
    )

optimizer = torch.optim.Adam(
    model.parameters(), args.LR, weight_decay=args.WEIGHT_DECAY
)
loaders = {
    "train": dataloader_train,
    "valid": dataloader_test,
}
runner = scRNARunner()

# Train model
runner.train(
    model=model,
    optimizer=optimizer,
    loaders=loaders,
    num_epochs=args.N_EPOCHS,
    callbacks=[
        dl.OptimizerCallback(metric_key="loss"),
        dl.CheckpointCallback(
            logdir=trial_dump_folder,
            loader_key="valid",
            metric_key="avg_acc",
            minimize=False,
            topk=1,
        ),
        dl.EarlyStoppingCallback(
            patience=150,
            loader_key="valid",
            metric_key="avg_acc",
            minimize=False,
            min_delta=1e-5,
        ),
        dl.ControlFlowCallbackWrapper(
            base_callback=dl.LoaderMetricCallback(
                metric=CustomMetric(),
                input_key=["embeddings_first", "embeddings_second", "temperature"],
                target_key=["embeddings_second"],
            ),
            ignore_loaders="train"  # Compute metrics only for validation, takes a long time on the training set
        )

    ],
    verbose=True,
)
