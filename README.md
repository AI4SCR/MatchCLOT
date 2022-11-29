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
