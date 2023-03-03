# Quickstart
**MatchCLOT** is a computational framework that is able to match single-cells measured using different omic modalities. With paired multi-omic data, MatchCLOT uses contrastive learning to learn a common representation between two modalities after applying a preprocessing pipeline for normalization and dimensionality reduction. Based on the similarities between the cells in the learned representation, MatchCLOT finds a matching between the cell profiles in the two omic modalities using entropic optimal transport. Pretrained MatchCLOT models can be applied to new unpaired multiomic data to match two modalities at single-cell resolution.

### Requirements
To use MatchCLOT for single-cell multimodal matching, you need two datasets with single-cell omic measurements in two different modalities (e.g. ATAC + GEX or GEX + ADT).

It is also possible to just use MatchCLOT as an embedding model for uni-modal data. In this case, you need a dataset with single-cell omic measurements in a single modality (e.g. ATAC, GEX or ADT).

The datasets can be provided in the form of a 'anndata.AnnData' object. The AnnData object should contain the following attributes:
- `obs`: a dataframe containing the cell annotations.
- `var`: a dataframe containing the feature annotations.
- `X`: a (sparse) matrix containing the cell-by-feature count matrix.

Refer to the [AnnData documentation](https://anndata.readthedocs.io/en/latest/) for more information.


```python
import matchclot

print(matchclot.__version__)
```

    0.1.0.dev1


### Download pretrained MatchCLOT
MatchCLOT comes with pretrained models for matching GEX and ATAC data. For this tutorial, we will use the model trained on the train set of the competition dataset for matching GEX and ATAC data available for download [here](https://ibm.box.com/s/3qhv2usv4n3aif2v3hml5eu5mmko5jbi). The folder contains the following files:
- `lsi_ATAC_transformer.pickle`: the pretrained preprocessing LSI for ATAC data
- `lsi_GEX_transformer.pickle`: the pretrained preprocessing LSI for GEX data
- `0/model.best.pth`: the pretrained contrastive learning model

These files should be placed in the `pretrain/GEX2ATAC` folder. inside the current working directory.

### Example Data
For this tutorial, we will use a subset of the test set of the [NeurIPS 2021 Single-Cell Competition Dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122), which contains 100 single-cell ATAC-seq and GEX measurements from human BMMCs.

### Run pretrained MatchCLOT
To run MatchCLOT, we provide the following arguments:
- `--PRETRAIN`: the path to the pretrained model folder
- `--CUSTOM_DATASET_PATH`: the path to the folder containing the dataset
- `--TRANSDUCTIVE`: whether to perform transductive preprocessing or not. For this tutorial, we will use `False`, since we only have access to the test set
- `--OUT`: the output name of the results
- `GEX2ATAC`: the type of matching to perform, `GEX2ATAC` means that we want to match GEX and ATAC data



```python
matchclot.run.main(["--PRETRAIN=pretrain", "--CUSTOM_DATASET_PATH=datasets/GEX_ATAC_quickstart/", "--TRANSDUCTIVE=False", "--OUT=quickstart_results", "GEX2ATAC"])
```

    Using device: cuda
    args: Namespace(TASK='GEX2ATAC', DATASETS_PATH='datasets', PRETRAIN_PATH='pretrain', OUT_NAME='quickstart_results', SCORES_PATH='scores', VALID_FOLD=0, HYPERPARAMS=True, OT_MATCHING=True, BATCH_LABEL_MATCHING=True, OT_ENTROPY=0.01, TRANSDUCTIVE=False, HARMONY=True, CUSTOM_DATASET_PATH='datasets/GEX_ATAC_quickstart/', SEED=0, SAVE_EMBEDDINGS=False, LR=0.0006, WEIGHT_DECAY=0.000125, EMBEDDING_DIM=128, DROPOUT_RATES_ATAC=0.67, DROPOUT_RATES_GEX0=0.34, DROPOUT_RATES_GEX1=0.47, LAYERS_DIM_ATAC=2048, LAYERS_DIM_GEX0=2048, LAYERS_DIM_GEX1=1024, LOG_T=2.74, N_LSI_COMPONENTS_ATAC=256, N_LSI_COMPONENTS_GEX=192, N_EPOCHS=7000, BATCH_SIZE=16384, SFA_NOISE=0.0) unknown_args: []
    mod1 cells: 100 mod2 cells: 100
    Use GPU mode.
    	Initialization is completed.
    	Completed 1 / 10 iteration(s).
    	Completed 2 / 10 iteration(s).
    Reach convergence after 2 iteration(s).
    Use GPU mode.
    	Initialization is completed.
    	Completed 1 / 10 iteration(s).
    	Completed 2 / 10 iteration(s).
    Reach convergence after 2 iteration(s).
    Loading weights from pretrain/GEX2ATAC/0/model.best.pth
    dropout list [Dropout(p=0.67, inplace=False)]
    SFA with noise: 0.0
    dropout list [Dropout(p=0.34, inplace=False), Dropout(p=0.47, inplace=False)]
    SFA with noise: 0.0
    batch label splits {'s4d1'}
    matching split s4d1
    It.  |Err
    -------------------
        0|4.303428e-02|
     4990|4.572171e-06|
    Prediction saved to pretrain/quickstart_resultsGEX2ATAC.h5ad
    For evaluation assuming cell order is the same in the two modalities
    top1 forward acc: 0.7200000286102295
    top1 backward acc: 0.7400000095367432
    top1 avg acc: 0.7300000190734863
    top5 forward acc: 0.9900000095367432
    top5 backward acc: 1.0
    top5 avg acc: 0.9950000047683716
    top1 competition metric: 0.6359237432479858
    top1 competition metric for soft predictions: 0.7200000286102295
    foscttm: 0.00419999985024333 foscttm_x: 0.00419999985024333 foscttm_y: 0.004200000315904617


    /home/ico/Documents/MatchCLOT-dev/lib/python3.10/site-packages/ot/bregman.py:517: UserWarning: Sinkhorn did not converge. You might want to increase the number of iterations `numItermax` or the regularization parameter `reg`.
      warnings.warn("Sinkhorn did not converge. You might want to "
    /home/ico/PycharmProjects/MatchCLOT-dev/matchclot/run/run.py:298: FutureWarning: X.dtype being converted to np.float32 from float64. In the next version of anndata (0.9) conversion will not be automatic. Pass dtype explicitly to avoid this warning. Pass `AnnData(X, dtype=X.dtype, ...)` to get the future behavour.
      out = ad.AnnData(


### Results on the example dataset
- **Top-1 accuracy: 0.73**, which means that 73% of the cell profiles were correctly matched
- **Top-5 accuracy: 0.995**, which means that 99.5% of the cell profiles have the correct match in the top-5 predicted matching scores
- **Top-1 competition metric for soft predictions** (since we are using entropic regularization for the OT matching): **0.72**, which is the metric used for the NeurIPS 2021 Single-Cell Competition, it is the average of the predicted matching scores corresponding to the correct matches
- **FOSCTTM: 0.0042**, fraction of samples closer than the true match (lower is better), which means that on average, for a cell profile, only 0.42% of the predicted matching scores are higher than the true matching score



```python

```
