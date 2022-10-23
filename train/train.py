import argparse
import os
import pickle
import sys

import anndata as ad
import pandas as pd
import torch
from catalyst import dl
from harmony import harmonize
from sklearn.model_selection import LeaveOneGroupOut

sys.path.append("../resources")
from resources.data import ModalityMatchingDataset
from resources.models import Modality_CLIP, Encoder, defaults_GEX2ADT, defaults_GEX2ATAC
from resources.catalyst_tools import scRNARunner, CustomMetric
from resources.preprocessing import lsiTransformer

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Define argument parsers
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest='TASK')

# Common args
defaults = dict()
for key, value in defaults.items():
    parser.add_argument("--" + key, default=value, type=type(value))

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
    train_path = "../datasets/phase2-data/match_modality/openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_"
    test_path = "../datasets/phase2-private-data/match_modality/openproblems_bmmc_cite_phase2_rna/openproblems_bmmc_cite_phase2_rna.censor_dataset.output_"
    pretrain_path = "../pretrainGEX2ADT"
    is_multiome = False
elif args.TASK == 'GEX2ATAC':
    train_path = "../datasets/phase2-data/match_modality/openproblems_bmmc_multiome_phase2_rna/openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_"
    test_path = "../datasets/phase2-private-data/match_modality/openproblems_bmmc_multiome_phase2_rna/openproblems_bmmc_multiome_phase2_rna.censor_dataset.output_"
    pretrain_path = "../pretrainGEX2ATAC"
    is_multiome = True
else:
    raise ValueError('Unknown task: ' + args.TASK)

par = {
    "input_train_mod1": f"{train_path}train_mod1.h5ad",
    "input_train_mod2": f"{train_path}train_mod2.h5ad",
    "input_train_sol": f"{train_path}train_sol.h5ad",
    "input_test_mod1": f"{test_path}test_mod1.h5ad",
    "input_test_mod2": f"{test_path}test_mod2.h5ad",
    "output_pretrain": pretrain_path,
    "input_pretrain": pretrain_path,
}
os.makedirs(par["output_pretrain"], exist_ok=True)

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
fold_number = int(args.VALID_FOLD)
print("fold_number:", fold_number)
trial_dump_folder = os.path.join(par["output_pretrain"], str(fold_number))
logo = LeaveOneGroupOut()
groups = sol_train.obs.batch
print("GROUPS:", groups)
logo.get_n_splits(input_train_mod2, groups=groups)
all_splits = list(logo.split(input_train_mod2, groups=groups))
train_indexes, test_indexes = all_splits[fold_number]

# Load or fit LSI preprocessing
path = par["output_pretrain"]
if os.path.exists(path + "/lsi_GEX_transformer.pickle"):
    # Avoid re-computing LSI transformation when using cross-validation and transductive LSI
    print("loading lsi transformer from", path)
    with open(path + "/lsi_GEX_transformer.pickle", "rb") as f:
        lsi_transformer_gex = pickle.load(f)
    if is_multiome:  # LSI is applied only on GEX and ATAC, not on ADT
        with open(path + "/lsi_ATAC_transformer.pickle", "rb") as f:
            lsi_transformer_atac = pickle.load(f)
else:
    print("No lsi transformer found in", path, "creating new one")
    os.makedirs(path, exist_ok=True)

    # Fit GEX LSI
    lsi_transformer_gex = lsiTransformer(
        n_components=args.N_LSI_COMPONENTS_GEX, drop_first=True
    )
    print("concatenating gex train and test")
    concatenated_gex = ad.concat([input_train_mod1, input_test_mod1], join="outer")
    print("done, concatenated_gex.shape", concatenated_gex.shape)
    lsi_transformer_gex.fit(concatenated_gex)
    # Save LSI transformation
    with open(path + "/lsi_GEX_transformer.pickle", "wb") as f:
        pickle.dump(lsi_transformer_gex, f)
    print("saved lsi pickle in ", path + "/lsi_GEX_transformer.pickle")

    # LSI is applied only on GEX and ATAC, not on ADT
    if is_multiome:
        # Fit ATAC LSI
        lsi_transformer_atac = lsiTransformer(
            n_components=args.N_LSI_COMPONENTS_ATAC, drop_first=True
        )
        print("concatenating atac train and test")
        concatenated_atac = ad.concat([input_train_mod2, input_test_mod2], join="outer")
        print("done, concatenated_atac.shape", concatenated_atac.shape)
        lsi_transformer_atac.fit(concatenated_atac)
        # Save LSI transformation
        with open(path + "/lsi_ATAC_transformer.pickle", "wb") as f:
            pickle.dump(lsi_transformer_atac, f)
        print("saved lsi pickle in ", path + "/lsi_ATAC_transformer.pickle")

# Apply LSI preprocessing
gex_train = lsi_transformer_gex.transform(input_train_mod1[train_indexes])
gex_test = lsi_transformer_gex.transform(input_train_mod1[test_indexes])
gex_private = lsi_transformer_gex.transform(input_test_mod1)
if is_multiome:
    atac_train = lsi_transformer_atac.transform(input_train_mod2[train_indexes])
    atac_test = lsi_transformer_atac.transform(input_train_mod2[test_indexes])
    atac_private = lsi_transformer_atac.transform(input_test_mod2)
else:
    adt_train = input_train_mod2[train_indexes].to_df()
    adt_test = input_train_mod2[test_indexes].to_df()
    adt_private = input_test_mod2.to_df()

# Apply Harmony batch effect correction, transductive setting
gex_train['batch'] = input_train_mod1.obs.batch[train_indexes]
gex_test['batch'] = input_train_mod1.obs.batch[test_indexes]
gex_private['batch'] = input_test_mod1.obs.batch
if is_multiome:
    atac_train['batch'] = input_train_mod2.obs.batch[train_indexes]
    atac_test['batch'] = input_train_mod2.obs.batch[test_indexes]
    atac_private['batch'] = input_test_mod2.obs.batch
else:
    adt_train['batch'] = input_train_mod2.obs.batch[train_indexes]
    adt_test['batch'] = input_train_mod2.obs.batch[test_indexes]
    adt_private['batch'] = input_test_mod2.obs.batch

def harmony_transductive(train, test, private, use_gpu=True):
    all = pd.concat([train, test, private])
    all_batches = all.pop('batch')
    all_batches.columns = ['batch']
    all_batches = all_batches.to_frame()
    all_harmony = harmonize(all.to_numpy(), all_batches, batch_key='batch', use_gpu=use_gpu, verbose=True)
    train_harmony = all_harmony[:len(train)]
    test_harmony = all_harmony[len(train):len(train) + len(test)]
    private_harmony = all_harmony[len(train) + len(test):]
    return train_harmony, test_harmony, private_harmony

gex_train_h, gex_test_h, gex_private_h = harmony_transductive(gex_train, gex_test, gex_private)
if is_multiome:
    # mod2 is ATAC
    mod2_train_h, mod2_test_h, mod2_private_h = harmony_transductive(atac_train, atac_test, atac_private)
else:
    # mod2 is ADT
    mod2_train_h, mod2_test_h, mod2_private_h = harmony_transductive(adt_train, adt_test, adt_private)

# Load torch dataloaders
dataset_train = ModalityMatchingDataset(pd.DataFrame(mod2_train_h), pd.DataFrame(gex_train_h))
dataset_test = ModalityMatchingDataset(pd.DataFrame(mod2_test_h), pd.DataFrame(gex_test_h))
dataset_private = ModalityMatchingDataset(pd.DataFrame(mod2_private_h), pd.DataFrame(gex_private_h))
dataloader_train = torch.utils.data.DataLoader(
    dataset_train, args.BATCH_SIZE, shuffle=True, num_workers=4
)
dataloader_test = torch.utils.data.DataLoader(
    dataset_test, args.BATCH_SIZE, shuffle=False, num_workers=4
)
dataloader_private = torch.utils.data.DataLoader(
    dataset_private, args.BATCH_SIZE, shuffle=False, num_workers=4
)
print("loaded dataloaders")

# Define modality encoders and trainer
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
    )
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
        dl.LoaderMetricCallback(
            metric=CustomMetric(),
            input_key=["embeddings_first", "embeddings_second", "temperature"],
            target_key=["embeddings_second"],
        ),
    ],
    verbose=True,
)