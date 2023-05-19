# MatchCLOT
Matching single cells across modalities with contrastive learning and optimal transport
## Required packages
- Python (tested with 3.8)
- Packages in [requirements.txt](requirements.txt) (tested with Virtualenv and the exact versions listed there)

## Dataset
If not already downloaded:
1) Install aws CLI, requires Python 3.8+ https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
2) `cd /path/to/MatchCLOT`
3) Download the 7.9 GiB dataset in the `datasets` folder: `aws s3 sync s3://openproblems-bio/public/phase2-private-data/match_modality/ ./datasets/ --no-sign-request`, this folder contains the phase 2 training data and the private test set data (with ground truth)


## Training
1) activate the virtual environment with the packages from [requirements.txt](requirements.txt)
2) `cd /path/to/MatchCLOT`

Run the following command from the [MatchCLOT](MatchCLOT) folder to train the model with default parameters:
- GEX2ADT
```
python train/train.py --VALID_FOLD=0 GEX2ADT
python train/train.py --VALID_FOLD=1 GEX2ADT
python train/train.py --VALID_FOLD=2 GEX2ADT
python train/train.py --VALID_FOLD=3 GEX2ADT
python train/train.py --VALID_FOLD=4 GEX2ADT
python train/train.py --VALID_FOLD=5 GEX2ADT
python train/train.py --VALID_FOLD=6 GEX2ADT
python train/train.py --VALID_FOLD=7 GEX2ADT
python train/train.py --VALID_FOLD=8 GEX2ADT
```
- GEX2ATAC
```
python train/train.py --VALID_FOLD=0 GEX2ATAC
```
## Evaluation
- GEX2ADT
  `python run/run.py --OUT="default" GEX2ADT`
- GEX2ATAC
  `python run/run.py --OUT="default" GEX2ATAC`

## Inference with pretrained models
1) Download the pretrained models from here: [GEX2ADT](https://ibm.box.com/s/bwcdwqpi9x662p7irklq41699h88fr59) [GEX2ATAC](https://ibm.box.com/s/3qhv2usv4n3aif2v3hml5eu5mmko5jbi)
2) Unzip the downloaded files in the [MatchCLOT](MatchCLOT) folder
3) Run the following command from the [MatchCLOT](MatchCLOT) folder:
```
python run/run.py --OUT="pbmc1" --B=False --T=False --HA=False --P=pretrainNoHA --CUSTOM_DATASET_PATH=datasets/PBMC/glue_processed/ GEX2ATAC
```
For example, this command will run the pretrained model on the dataset in `datasets/PBMC/glue_processed/`.
The dataset should be composed of 2 files: `test_mod1.h5ad` and `test_mod2.h5ad`,
where test_mod1 is the GEX dataset and test_mod2 is the ATAC or ADT dataset.
The `--B=False` flag disables the batch label matching and is used when the dataset does not have batch labels or is composed of a single batch.
The `--T=False` flag disables the transductive preprocessing steps and is used when testing on a dataset not available during training.
The `--HA=False` flag disables the Harmony batch effect correction step and is used when `--B=False`.
The `--P=pretrainNoHA` flag specifies the pretrained model to use. The `--CUSTOM_DATASET_PATH=datasets/PBMC/glue_processed/` flag specifies the dataset to use. The `GEX2ATAC` flag specifies the task to run.

## Ablation study
### No improved hyperparameters
  - GEX2ADT
    * Train
      ```
      python train/train.py --P=pretrainNoHY --HY=False --V=0 GEX2ADT
      python train/train.py --P=pretrainNoHY --HY=False --V=1 GEX2ADT
      python train/train.py --P=pretrainNoHY --HY=False --V=2 GEX2ADT
      python train/train.py --P=pretrainNoHY --HY=False --V=3 GEX2ADT
      python train/train.py --P=pretrainNoHY --HY=False --V=4 GEX2ADT
      python train/train.py --P=pretrainNoHY --HY=False --V=5 GEX2ADT
      python train/train.py --P=pretrainNoHY --HY=False --V=6 GEX2ADT
      python train/train.py --P=pretrainNoHY --HY=False --V=7 GEX2ADT
      python train/train.py --P=pretrainNoHY --HY=False --V=8 GEX2ADT
      ```
    * Evaluate
      ```
      python run/run.py --OUT="NoHY" --P=pretrainNoHY --HY=False GEX2ADT
      ```
    
  - GEX2ATAC
    * Train
      ```
      python train/train.py --P=pretrainNoHY --HY=False --V=0 GEX2ATAC
      ```
    * Evaluate
      ```
      python run/run.py --OUT="NoHY" --P=pretrainNoHY --HY=False GEX2ATAC
      ```

### No OT matching 
  Does not require retraining
  - GEX2ADT
    ```
    python run/run.py --OUT="NoOT" --P=pretrain --OT_M=False GEX2ADT
    ```
  - GEX2ATAC
    ```
    python run/run.py --OUT="NoOT" --P=pretrain --OT_M=False GEX2ATAC
    ```

### No batch label matching
  Does not require retraining
  - GEX2ADT
    ```
    python run/run.py --OUT="NoB" --P=pretrain --B=False GEX2ADT
    ```
  - GEX2ATAC
    ```
    python run/run.py --OUT="NoB" --P=pretrain --B=False GEX2ATAC
    ```

### No entropic regularization for OT matching
  Does not require retraining
  - GEX2ADT
    ```
    python run/run.py --OUT="NoE" --P=pretrain --OT_E=0.0 GEX2ADT
    ```
  - GEX2ATAC
    ```
    python run/run.py --OUT="NoE" --P=pretrain --OT_E=0.0 GEX2ATAC
    ```

### No transductive preprocessing
  - GEX2ADT
    * Train
      ```
      python train/train.py --P=pretrainNoT --T=False --V=0 GEX2ADT
      python train/train.py --P=pretrainNoT --T=False --V=1 GEX2ADT
      python train/train.py --P=pretrainNoT --T=False --V=2 GEX2ADT
      python train/train.py --P=pretrainNoT --T=False --V=3 GEX2ADT
      python train/train.py --P=pretrainNoT --T=False --V=4 GEX2ADT
      python train/train.py --P=pretrainNoT --T=False --V=5 GEX2ADT
      python train/train.py --P=pretrainNoT --T=False --V=6 GEX2ADT
      python train/train.py --P=pretrainNoT --T=False --V=7 GEX2ADT
      python train/train.py --P=pretrainNoT --T=False --V=8 GEX2ADT
      ```
    * Evaluate
      ```
      python run/run.py --OUT="NoT" --P=pretrainNoT --T=False GEX2ADT
      ```
  - GEX2ATAC
    * Train
      ```
      python train/train.py --P=pretrainNoT --T=False --V=0 GEX2ATAC
      ```
    * Evaluate
      ```
      python run/run.py --OUT="NoT" --P=pretrainNoT --T=False GEX2ATAC
      ```

### No Harmony preprocessing
  - GEX2ADT
    * Train
      ```
      python train/train.py --P=pretrainNoHA --HA=False --V=0 GEX2ADT
      python train/train.py --P=pretrainNoHA --HA=False --V=1 GEX2ADT
      python train/train.py --P=pretrainNoHA --HA=False --V=2 GEX2ADT
      python train/train.py --P=pretrainNoHA --HA=False --V=3 GEX2ADT
      python train/train.py --P=pretrainNoHA --HA=False --V=4 GEX2ADT
      python train/train.py --P=pretrainNoHA --HA=False --V=5 GEX2ADT
      python train/train.py --P=pretrainNoHA --HA=False --V=6 GEX2ADT
      python train/train.py --P=pretrainNoHA --HA=False --V=7 GEX2ADT
      python train/train.py --P=pretrainNoHA --HA=False --V=8 GEX2ADT
      ```
    * Evaluate
      ```
      python run/run.py --OUT="NoHA" --P=pretrainNoHA --HA=False GEX2ADT
      ```
  - GEX2ATAC
    * Train
      ```
      python train/train.py --P=pretrainNoHA --HA=False --V=0 GEX2ATAC
      ```
    * Evaluate
      ```
      python run/run.py --OUT="NoHA" --P=pretrainNoHA --HA=False GEX2ATAC
      ```
