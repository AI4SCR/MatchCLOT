# SCOOTR
SCOOTR: Single-Cell Multimodal Data Integration with Contrastive Learning and Optimal Transport

## Required packages
- Python (tested with 3.8)
- Packages in [requirements.txt](requirements.txt) (tested with Virtualenv and the exact versions listed there)

## Dataset
If not already downloaded:
1) Install aws CLI, requires Python 3.8+ https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
2) `cd /path/to/SCOOTR`
3) Download the 7.9 GiB dataset in the `datasets` folder: `aws s3 sync s3://openproblems-bio/public/phase2-private-data/match_modality/ ./datasets/ --no-sign-request`, this folder contains the phase 2 training data and the private test set data (with ground truth)


## Training
### On CCC
1) `ssh username@cccxl010.pok.ibm.com `
2) activate the virtual environment with the packages from [requirements.txt](requirements.txt)
3) `cd /path/to/SCOOTR`

Run the following command from the [SCOOTR](SCOOTR) folder to train the model with default parameters (CCC commands):
- GEX2ADT
```
jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --VALID_FOLD=0 GEX2ADT
jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --VALID_FOLD=1 GEX2ADT
jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --VALID_FOLD=2 GEX2ADT
jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --VALID_FOLD=3 GEX2ADT
jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --VALID_FOLD=4 GEX2ADT
jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --VALID_FOLD=5 GEX2ADT
jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --VALID_FOLD=6 GEX2ADT
jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --VALID_FOLD=7 GEX2ADT
jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --VALID_FOLD=8 GEX2ADT
```
- GEX2ATAC
```
jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --VALID_FOLD=0 GEX2ATAC
```
## Evaluation
- GEX2ADT
1) `jbsub -mem 50g -cores 4+1 -q x86_1h python run/run.py GEX2ADT`
2) `jbsub -mem 50g -cores 4 -q x86_1h python run/evaluate.py -p=pretrain/GEX2ADT.h5ad GEX2ADT`
- GEX2ATAC
1) `jbsub -mem 50g -cores 4+1 -q x86_1h python run/run.py GEX2ATAC`
2) `jbsub -mem 50g -cores 4 -q x86_1h python run/evaluate.py -p=pretrain/GEX2ATAC.h5ad GEX2ATAC`

## Ablation study
### No improved hyperparameters
  - GEX2ADT
    * Train
      ```
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHY --HY=False --V=0 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHY --HY=False --V=1 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHY --HY=False --V=2 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHY --HY=False --V=3 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHY --HY=False --V=4 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHY --HY=False --V=5 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHY --HY=False --V=6 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHY --HY=False --V=7 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHY --HY=False --V=8 GEX2ADT
      ```
    * Evaluate
      ```
      jbsub -mem 50g -cores 4+1 -q x86_1h python run/run.py --P=pretrainNoHY --HY=False GEX2ADT
      jbsub -mem 50g -cores 4 -q x86_1h python run/evaluate.py -p=pretrainNoHY/GEX2ADT.h5ad GEX2ADT
      ```
    
  - GEX2ATAC
    * Train
      ```
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHY --HY=False --V=0 GEX2ATAC
      ```
    * Evaluate
      ```
      jbsub -mem 50g -cores 4+1 -q x86_1h python run/run.py --P=pretrainNoHY --HY=False GEX2ATAC
      jbsub -mem 50g -cores 4 -q x86_1h python run/evaluate.py -p=pretrainNoHY/GEX2ATAC.h5ad GEX2ATAC
      ```

### No OT matching 
  Does not require retraining
  - GEX2ADT
    ```
    jbsub -mem 50g -cores 4+1 -q x86_1h python run/run.py --P=pretrain --OT_M=False GEX2ADT
    jbsub -mem 50g -cores 4 -q x86_1h python run/evaluate.py -p=pretrain/GEX2ADT.h5ad GEX2ADT
    ```
  - GEX2ATAC
    ```
    jbsub -mem 50g -cores 4+1 -q x86_1h python run/run.py --P=pretrain --OT_M=False GEX2ATAC
    jbsub -mem 50g -cores 4 -q x86_1h python run/evaluate.py -p=pretrain/GEX2ATAC.h5ad GEX2ATAC
    ```

### No batch label matching
  Does not require retraining
  - GEX2ADT
    ```
    jbsub -mem 50g -cores 4+1 -q x86_1h python run/run.py --P=pretrain --B=False GEX2ADT
    jbsub -mem 50g -cores 4 -q x86_1h python run/evaluate.py -p=pretrain/GEX2ADT.h5ad GEX2ADT
    ```
  - GEX2ATAC
    ```
    jbsub -mem 50g -cores 4+1 -q x86_1h python run/run.py --P=pretrain --B=False GEX2ATAC
    jbsub -mem 50g -cores 4 -q x86_1h python run/evaluate.py -p=pretrain/GEX2ATAC.h5ad GEX2ATAC
    ```

### No entropic regularization for OT matching
  Does not require retraining
  - GEX2ADT
    ```
    jbsub -mem 50g -cores 4+1 -q x86_1h python run/run.py --P=pretrain --OT_E=0.0 GEX2ADT
    jbsub -mem 50g -cores 4 -q x86_1h python run/evaluate.py -p=pretrain/GEX2ADT.h5ad GEX2ADT
    ```
  - GEX2ATAC
    ```
    jbsub -mem 50g -cores 4+1 -q x86_1h python run/run.py --P=pretrain --OT_E=0.0 GEX2ATAC
    jbsub -mem 50g -cores 4 -q x86_1h python run/evaluate.py -p=pretrain/GEX2ATAC.h5ad GEX2ATAC
    ```

### No transductive preprocessing
  - GEX2ADT
    * Train
      ```
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoT --T=False --V=0 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoT --T=False --V=1 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoT --T=False --V=2 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoT --T=False --V=3 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoT --T=False --V=4 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoT --T=False --V=5 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoT --T=False --V=6 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoT --T=False --V=7 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoT --T=False --V=8 GEX2ADT
      ```
    * Evaluate
      ```
      jbsub -mem 50g -cores 4+1 -q x86_1h python run/run.py --P=pretrainNoT --T=False GEX2ADT
      jbsub -mem 50g -cores 4 -q x86_1h python run/evaluate.py -p=pretrainNoT/GEX2ADT.h5ad GEX2ADT
      ```
  - GEX2ATAC
    * Train
      ```
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoT --T=False --V=0 GEX2ATAC
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoT --T=False --V=1 GEX2ATAC
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoT --T=False --V=2 GEX2ATAC
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoT --T=False --V=3 GEX2ATAC
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoT --T=False --V=4 GEX2ATAC
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoT --T=False --V=5 GEX2ATAC
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoT --T=False --V=6 GEX2ATAC
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoT --T=False --V=7 GEX2ATAC
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoT --T=False --V=8 GEX2ATAC
      ```
    * Evaluate
      ```
      jbsub -mem 50g -cores 4+1 -q x86_1h python run/run.py --P=pretrainNoT --T=False GEX2ATAC
      jbsub -mem 50g -cores 4 -q x86_1h python run/evaluate.py -p=pretrainNoT/GEX2ATAC.h5ad GEX2ATAC
      ```

### No Harmony preprocessing
  - GEX2ADT
    * Train
      ```
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHA --HA=False --V=0 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHA --HA=False --V=1 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHA --HA=False --V=2 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHA --HA=False --V=3 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHA --HA=False --V=4 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHA --HA=False --V=5 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHA --HA=False --V=6 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHA --HA=False --V=7 GEX2ADT
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHA --HA=False --V=8 GEX2ADT
      ```
    * Evaluate
      ```
      jbsub -mem 50g -cores 4+1 -q x86_1h python run/run.py --P=pretrainNoHA --HA=False GEX2ADT
      jbsub -mem 50g -cores 4 -q x86_1h python run/evaluate.py -p=pretrainNoHA/GEX2ADT.h5ad GEX2ADT
      ```
  - GEX2ATAC
    * Train
      ```
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHA --HA=False --V=0 GEX2ATAC
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHA --HA=False --V=1 GEX2ATAC
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHA --HA=False --V=2 GEX2ATAC
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHA --HA=False --V=3 GEX2ATAC
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHA --HA=False --V=4 GEX2ATAC
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHA --HA=False --V=5 GEX2ATAC
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHA --HA=False --V=6 GEX2ATAC
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHA --HA=False --V=7 GEX2ATAC
      jbsub -mem 50g -cores 4+1 -q x86_24h python train/train.py --P=pretrainNoHA --HA=False --V=8 GEX2ATAC
      ```
    * Evaluate
      ```
      jbsub -mem 50g -cores 4+1 -q x86_1h python run/run.py --P=pretrainNoHA --HA=False GEX2ATAC
      jbsub -mem 50g -cores 4 -q x86_1h python run/evaluate.py -p=pretrainNoHA/GEX2ATAC.h5ad GEX2ATAC
      ```
      
