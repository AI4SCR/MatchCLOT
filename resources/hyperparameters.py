# Default args and hyperparameters
defaults_common = dict(
    DATASETS_PATH="datasets",
    PRETRAIN_PATH="pretrain",
    VALID_FOLD=0,  # validation on the first batch (s1d1), train on other batches
    HYPERPARAMS=True,
    OT_MATCHING=True,
    BATCH_LABEL_MATCHING=True,
    OT_ENTROPY=0.01,
    TRANSDUCTIVE=True,
    HARMONY=True,
)

defaults_GEX2ATAC = dict(
    LR=0.0006,
    WEIGHT_DECAY=0.000125,
    EMBEDDING_DIM=128,
    DROPOUT_RATES_ATAC=0.67,
    DROPOUT_RATES_GEX0=0.34,
    DROPOUT_RATES_GEX1=0.47,
    LAYERS_DIM_ATAC=2048,
    LAYERS_DIM_GEX0=2048,
    LAYERS_DIM_GEX1=1024,
    LOG_T=2.74,
    N_LSI_COMPONENTS_ATAC=256,
    N_LSI_COMPONENTS_GEX=192,
    N_EPOCHS=7000,
    BATCH_SIZE=16384,
    SFA_NOISE=0.0,
)

defaults_GEX2ADT = dict(
    LR=0.000175,
    WEIGHT_DECAY=0.0002,
    EMBEDDING_DIM=256,
    DROPOUT_RATES_ADT0=0.4,
    DROPOUT_RATES_ADT1=0.2,
    DROPOUT_RATES_GEX0=0.3,
    DROPOUT_RATES_GEX1=0.05,
    LAYERS_DIM_ADT0=4096,
    LAYERS_DIM_ADT1=2048,
    LAYERS_DIM_GEX0=256,
    LAYERS_DIM_GEX1=2048,
    LOG_T=4.0,
    N_LSI_COMPONENTS_GEX=192,
    N_LSI_COMPONENTS_ADT=134,  # ADT data is not processed with LSI, just used for the encoder input dimension
    N_EPOCHS=7000,
    BATCH_SIZE=16384,
    SFA_NOISE=0.0,
)

# Team Novel baseline hyperparameters
baseline_GEX2ATAC = dict(
    LR=0.000585,
    WEIGHT_DECAY=0.0,
    EMBEDDING_DIM=256,
    DROPOUT_RATES_ATAC=0.661,
    DROPOUT_RATES_GEX0=0.541,
    DROPOUT_RATES_GEX1=0.396,
    LAYERS_DIM_ATAC=2048,
    LAYERS_DIM_GEX0=1024,
    LAYERS_DIM_GEX1=1024,
    LOG_T=3.065016,
    N_LSI_COMPONENTS_ATAC=512,
    N_LSI_COMPONENTS_GEX=64,
    N_EPOCHS=7000,
    BATCH_SIZE=16384,
    SFA_NOISE=0.0,
)

baseline_GEX2ADT = dict(
    LR=7.79984e-05,
    WEIGHT_DECAY=0.0,
    EMBEDDING_DIM=64,
    DROPOUT_RATES_ADT0=0.0221735,
    DROPOUT_RATES_ADT1=0.296919,
    DROPOUT_RATES_GEX0=0.0107121,
    DROPOUT_RATES_GEX1=0.254689,
    LAYERS_DIM_ADT0=512,
    LAYERS_DIM_ADT1=2048,
    LAYERS_DIM_GEX0=1024,
    LAYERS_DIM_GEX1=512,
    LOG_T=3.463735,
    N_LSI_COMPONENTS_GEX=128,
    N_LSI_COMPONENTS_ADT=134,  # ADT data is not processed with LSI, just used for the encoder input dimension
    N_EPOCHS=7000,
    BATCH_SIZE=2048,
    SFA_NOISE=0.0,
)
