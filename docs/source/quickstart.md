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
       10|2.897871e-03|
       20|1.226706e-03|
       30|7.718077e-04|
       40|5.675808e-04|
       50|4.521399e-04|
       60|3.790998e-04|
       70|3.290487e-04|
       80|2.924361e-04|
       90|2.642403e-04|
      100|2.416716e-04|
      110|2.230797e-04|
      120|2.074243e-04|
      130|1.940126e-04|
      140|1.823619e-04|
      150|1.721237e-04|
      160|1.630394e-04|
      170|1.549125e-04|
      180|1.475902e-04|
      190|1.409523e-04|
    It.  |Err
    -------------------
      200|1.349022e-04|
      210|1.293614e-04|
      220|1.242654e-04|
      230|1.195607e-04|
      240|1.152022e-04|
      250|1.111519e-04|
      260|1.073773e-04|
      270|1.038507e-04|
      280|1.005479e-04|
      290|9.744808e-05|
      300|9.453291e-05|
      310|9.178637e-05|
      320|8.919430e-05|
      330|8.674416e-05|
      340|8.442479e-05|
      350|8.222622e-05|
      360|8.013950e-05|
      370|7.815658e-05|
      380|7.627021e-05|
      390|7.447378e-05|
    It.  |Err
    -------------------
      400|7.276134e-05|
      410|7.112742e-05|
      420|6.956704e-05|
      430|6.807564e-05|
      440|6.664903e-05|
      450|6.528336e-05|
      460|6.397506e-05|
      470|6.272085e-05|
      480|6.151768e-05|
      490|6.036271e-05|
      500|5.925332e-05|
      510|5.818707e-05|
      520|5.716166e-05|
      530|5.617496e-05|
      540|5.522498e-05|
      550|5.430985e-05|
      560|5.342781e-05|
      570|5.257722e-05|
      580|5.175653e-05|
      590|5.096428e-05|
    It.  |Err
    -------------------
      600|5.019911e-05|
      610|4.945972e-05|
      620|4.874490e-05|
      630|4.805348e-05|
      640|4.738440e-05|
      650|4.673661e-05|
      660|4.610914e-05|
      670|4.550108e-05|
      680|4.491156e-05|
      690|4.433974e-05|
      700|4.378484e-05|
      710|4.324612e-05|
      720|4.272288e-05|
      730|4.221445e-05|
      740|4.172019e-05|
      750|4.123949e-05|
      760|4.077179e-05|
      770|4.031654e-05|
      780|3.987322e-05|
      790|3.944133e-05|
    It.  |Err
    -------------------
      800|3.902042e-05|
      810|3.861002e-05|
      820|3.820972e-05|
      830|3.781911e-05|
      840|3.743781e-05|
      850|3.706545e-05|
      860|3.670168e-05|
      870|3.634616e-05|
      880|3.599859e-05|
      890|3.565865e-05|
      900|3.532606e-05|
      910|3.500054e-05|
      920|3.468183e-05|
      930|3.436968e-05|
      940|3.406385e-05|
      950|3.376411e-05|
      960|3.347025e-05|
      970|3.318206e-05|
      980|3.289933e-05|
      990|3.262188e-05|
    It.  |Err
    -------------------
     1000|3.234953e-05|
     1010|3.208210e-05|
     1020|3.181943e-05|
     1030|3.156135e-05|
     1040|3.130773e-05|
     1050|3.105841e-05|
     1060|3.081325e-05|
     1070|3.057212e-05|
     1080|3.033490e-05|
     1090|3.010146e-05|
     1100|2.987169e-05|
     1110|2.964548e-05|
     1120|2.942271e-05|
     1130|2.920328e-05|
     1140|2.898711e-05|
     1150|2.877409e-05|
     1160|2.856413e-05|
     1170|2.835715e-05|
     1180|2.815306e-05|
     1190|2.795178e-05|
    It.  |Err
    -------------------
     1200|2.775324e-05|
     1210|2.755737e-05|
     1220|2.736408e-05|
     1230|2.717332e-05|
     1240|2.698502e-05|
     1250|2.679912e-05|
     1260|2.661555e-05|
     1270|2.643425e-05|
     1280|2.625518e-05|
     1290|2.607828e-05|
     1300|2.590349e-05|
     1310|2.573076e-05|
     1320|2.556006e-05|
     1330|2.539132e-05|
     1340|2.522451e-05|
     1350|2.505958e-05|
     1360|2.489649e-05|
     1370|2.473520e-05|
     1380|2.457567e-05|
     1390|2.441787e-05|
    It.  |Err
    -------------------
     1400|2.426175e-05|
     1410|2.410728e-05|
     1420|2.395444e-05|
     1430|2.380318e-05|
     1440|2.365347e-05|
     1450|2.350529e-05|
     1460|2.335860e-05|
     1470|2.321338e-05|
     1480|2.306959e-05|
     1490|2.292721e-05|
     1500|2.278622e-05|
     1510|2.264659e-05|
     1520|2.250829e-05|
     1530|2.237130e-05|
     1540|2.223560e-05|
     1550|2.210116e-05|
     1560|2.196797e-05|
     1570|2.183600e-05|
     1580|2.170523e-05|
     1590|2.157564e-05|
    It.  |Err
    -------------------
     1600|2.144722e-05|
     1610|2.131993e-05|
     1620|2.119378e-05|
     1630|2.106873e-05|
     1640|2.094477e-05|
     1650|2.082188e-05|
     1660|2.070005e-05|
     1670|2.057926e-05|
     1680|2.045950e-05|
     1690|2.034075e-05|
     1700|2.022299e-05|
     1710|2.010621e-05|
     1720|1.999040e-05|
     1730|1.987555e-05|
     1740|1.976163e-05|
     1750|1.964864e-05|
     1760|1.953657e-05|
     1770|1.942540e-05|
     1780|1.931511e-05|
     1790|1.920571e-05|
    It.  |Err
    -------------------
     1800|1.909717e-05|
     1810|1.898949e-05|
     1820|1.888265e-05|
     1830|1.877664e-05|
     1840|1.867146e-05|
     1850|1.856709e-05|
     1860|1.846352e-05|
     1870|1.836075e-05|
     1880|1.825876e-05|
     1890|1.815754e-05|
     1900|1.805709e-05|
     1910|1.795739e-05|
     1920|1.785844e-05|
     1930|1.776022e-05|
     1940|1.766274e-05|
     1950|1.756597e-05|
     1960|1.746992e-05|
     1970|1.737457e-05|
     1980|1.727992e-05|
     1990|1.718596e-05|
    It.  |Err
    -------------------
     2000|1.709268e-05|
     2010|1.700008e-05|
     2020|1.690814e-05|
     2030|1.681686e-05|
     2040|1.672624e-05|
     2050|1.663626e-05|
     2060|1.654692e-05|
     2070|1.645821e-05|
     2080|1.637013e-05|
     2090|1.628266e-05|
     2100|1.619581e-05|
     2110|1.610957e-05|
     2120|1.602393e-05|
     2130|1.593889e-05|
     2140|1.585443e-05|
     2150|1.577055e-05|
     2160|1.568726e-05|
     2170|1.560454e-05|
     2180|1.552238e-05|
     2190|1.544078e-05|
    It.  |Err
    -------------------
     2200|1.535974e-05|
     2210|1.527925e-05|
     2220|1.519931e-05|
     2230|1.511991e-05|
     2240|1.504104e-05|
     2250|1.496270e-05|
     2260|1.488489e-05|
     2270|1.480760e-05|
     2280|1.473083e-05|
     2290|1.465456e-05|
     2300|1.457881e-05|
     2310|1.450355e-05|
     2320|1.442880e-05|
     2330|1.435454e-05|
     2340|1.428076e-05|
     2350|1.420747e-05|
     2360|1.413467e-05|
     2370|1.406234e-05|
     2380|1.399048e-05|
     2390|1.391909e-05|
    It.  |Err
    -------------------
     2400|1.384817e-05|
     2410|1.377770e-05|
     2420|1.370769e-05|
     2430|1.363814e-05|
     2440|1.356904e-05|
     2450|1.350038e-05|
     2460|1.343216e-05|
     2470|1.336438e-05|
     2480|1.329704e-05|
     2490|1.323012e-05|
     2500|1.316364e-05|
     2510|1.309757e-05|
     2520|1.303193e-05|
     2530|1.296671e-05|
     2540|1.290190e-05|
     2550|1.283750e-05|
     2560|1.277351e-05|
     2570|1.270992e-05|
     2580|1.264674e-05|
     2590|1.258395e-05|
    It.  |Err
    -------------------
     2600|1.252156e-05|
     2610|1.245956e-05|
     2620|1.239794e-05|
     2630|1.233672e-05|
     2640|1.227587e-05|
     2650|1.221541e-05|
     2660|1.215532e-05|
     2670|1.209561e-05|
     2680|1.203627e-05|
     2690|1.197730e-05|
     2700|1.191869e-05|
     2710|1.186045e-05|
     2720|1.180257e-05|
     2730|1.174504e-05|
     2740|1.168787e-05|
     2750|1.163105e-05|
     2760|1.157458e-05|
     2770|1.151846e-05|
     2780|1.146269e-05|
     2790|1.140725e-05|
    It.  |Err
    -------------------
     2800|1.135216e-05|
     2810|1.129740e-05|
     2820|1.124298e-05|
     2830|1.118889e-05|
     2840|1.113513e-05|
     2850|1.108170e-05|
     2860|1.102859e-05|
     2870|1.097581e-05|
     2880|1.092335e-05|
     2890|1.087120e-05|
     2900|1.081937e-05|
     2910|1.076786e-05|
     2920|1.071665e-05|
     2930|1.066576e-05|
     2940|1.061518e-05|
     2950|1.056489e-05|
     2960|1.051492e-05|
     2970|1.046524e-05|
     2980|1.041586e-05|
     2990|1.036678e-05|
    It.  |Err
    -------------------
     3000|1.031799e-05|
     3010|1.026950e-05|
     3020|1.022130e-05|
     3030|1.017338e-05|
     3040|1.012575e-05|
     3050|1.007841e-05|
     3060|1.003135e-05|
     3070|9.984570e-06|
     3080|9.938070e-06|
     3090|9.891846e-06|
     3100|9.845898e-06|
     3110|9.800223e-06|
     3120|9.754820e-06|
     3130|9.709686e-06|
     3140|9.664820e-06|
     3150|9.620220e-06|
     3160|9.575885e-06|
     3170|9.531811e-06|
     3180|9.487999e-06|
     3190|9.444446e-06|
    It.  |Err
    -------------------
     3200|9.401150e-06|
     3210|9.358109e-06|
     3220|9.315323e-06|
     3230|9.272789e-06|
     3240|9.230505e-06|
     3250|9.188470e-06|
     3260|9.146682e-06|
     3270|9.105140e-06|
     3280|9.063842e-06|
     3290|9.022787e-06|
     3300|8.981972e-06|
     3310|8.941397e-06|
     3320|8.901059e-06|
     3330|8.860957e-06|
     3340|8.821090e-06|
     3350|8.781457e-06|
     3360|8.742054e-06|
     3370|8.702882e-06|
     3380|8.663938e-06|
     3390|8.625222e-06|
    It.  |Err
    -------------------
     3400|8.586731e-06|
     3410|8.548464e-06|
     3420|8.510420e-06|
     3430|8.472598e-06|
     3440|8.434995e-06|
     3450|8.397611e-06|
     3460|8.360444e-06|
     3470|8.323492e-06|
     3480|8.286755e-06|
     3490|8.250231e-06|
     3500|8.213919e-06|
     3510|8.177816e-06|
     3520|8.141923e-06|
     3530|8.106237e-06|
     3540|8.070758e-06|
     3550|8.035483e-06|
     3560|8.000412e-06|
     3570|7.965544e-06|
     3580|7.930876e-06|
     3590|7.896409e-06|
    It.  |Err
    -------------------
     3600|7.862140e-06|
     3610|7.828068e-06|
     3620|7.794192e-06|
     3630|7.760512e-06|
     3640|7.727024e-06|
     3650|7.693730e-06|
     3660|7.660626e-06|
     3670|7.627712e-06|
     3680|7.594987e-06|
     3690|7.562450e-06|
     3700|7.530100e-06|
     3710|7.497934e-06|
     3720|7.465953e-06|
     3730|7.434154e-06|
     3740|7.402538e-06|
     3750|7.371102e-06|
     3760|7.339846e-06|
     3770|7.308768e-06|
     3780|7.277868e-06|
     3790|7.247143e-06|
    It.  |Err
    -------------------
     3800|7.216594e-06|
     3810|7.186219e-06|
     3820|7.156018e-06|
     3830|7.125988e-06|
     3840|7.096128e-06|
     3850|7.066439e-06|
     3860|7.036919e-06|
     3870|7.007566e-06|
     3880|6.978380e-06|
     3890|6.949359e-06|
     3900|6.920503e-06|
     3910|6.891811e-06|
     3920|6.863281e-06|
     3930|6.834913e-06|
     3940|6.806706e-06|
     3950|6.778658e-06|
     3960|6.750769e-06|
     3970|6.723038e-06|
     3980|6.695463e-06|
     3990|6.668044e-06|
    It.  |Err
    -------------------
     4000|6.640780e-06|
     4010|6.613670e-06|
     4020|6.586712e-06|
     4030|6.559907e-06|
     4040|6.533253e-06|
     4050|6.506748e-06|
     4060|6.480393e-06|
     4070|6.454187e-06|
     4080|6.428127e-06|
     4090|6.402214e-06|
     4100|6.376447e-06|
     4110|6.350825e-06|
     4120|6.325346e-06|
     4130|6.300010e-06|
     4140|6.274817e-06|
     4150|6.249764e-06|
     4160|6.224852e-06|
     4170|6.200080e-06|
     4180|6.175446e-06|
     4190|6.150950e-06|
    It.  |Err
    -------------------
     4200|6.126591e-06|
     4210|6.102368e-06|
     4220|6.078280e-06|
     4230|6.054327e-06|
     4240|6.030508e-06|
     4250|6.006821e-06|
     4260|5.983267e-06|
     4270|5.959844e-06|
     4280|5.936551e-06|
     4290|5.913388e-06|
     4300|5.890354e-06|
     4310|5.867448e-06|
     4320|5.844670e-06|
     4330|5.822018e-06|
     4340|5.799492e-06|
     4350|5.777091e-06|
     4360|5.754814e-06|
     4370|5.732661e-06|
     4380|5.710630e-06|
     4390|5.688722e-06|
    It.  |Err
    -------------------
     4400|5.666935e-06|
     4410|5.645269e-06|
     4420|5.623722e-06|
     4430|5.602295e-06|
     4440|5.580986e-06|
     4450|5.559795e-06|
     4460|5.538720e-06|
     4470|5.517763e-06|
     4480|5.496920e-06|
     4490|5.476193e-06|
     4500|5.455580e-06|
     4510|5.435080e-06|
     4520|5.414694e-06|
     4530|5.394419e-06|
     4540|5.374256e-06|
     4550|5.354204e-06|
     4560|5.334262e-06|
     4570|5.314430e-06|
     4580|5.294706e-06|
     4590|5.275091e-06|
    It.  |Err
    -------------------
     4600|5.255583e-06|
     4610|5.236182e-06|
     4620|5.216887e-06|
     4630|5.197698e-06|
     4640|5.178614e-06|
     4650|5.159634e-06|
     4660|5.140758e-06|
     4670|5.121985e-06|
     4680|5.103315e-06|
     4690|5.084746e-06|
     4700|5.066279e-06|
     4710|5.047913e-06|
     4720|5.029646e-06|
     4730|5.011479e-06|
     4740|4.993411e-06|
     4750|4.975441e-06|
     4760|4.957569e-06|
     4770|4.939794e-06|
     4780|4.922115e-06|
     4790|4.904533e-06|
    It.  |Err
    -------------------
     4800|4.887045e-06|
     4810|4.869653e-06|
     4820|4.852355e-06|
     4830|4.835150e-06|
     4840|4.818039e-06|
     4850|4.801020e-06|
     4860|4.784093e-06|
     4870|4.767258e-06|
     4880|4.750513e-06|
     4890|4.733859e-06|
     4900|4.717295e-06|
     4910|4.700820e-06|
     4920|4.684434e-06|
     4930|4.668136e-06|
     4940|4.651925e-06|
     4950|4.635802e-06|
     4960|4.619766e-06|
     4970|4.603815e-06|
     4980|4.587951e-06|
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
