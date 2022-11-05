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
4) `jbsub -mem 50g -cores 4+1 -q x86_24h` followed by the python commands

Run the following command from the [SCOOTR](SCOOTR) folder to train the model with default parameters:
- `python train/train.py --VALID_FOLD=0 GEX2ADT`
- `python train/train.py --VALID_FOLD=0 GEX2ATAC`

Cross-validation training (CCC commands):
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
2) `jbsub -mem 50g -cores 4 -q x86_1h python run/evaluate.py -p=GEX2ADT.h5ad GEX2ADT`
- GEX2ATAC
1) `jbsub -mem 50g -cores 4+1 -q x86_1h python run/run.py GEX2ATAC`
2) `jbsub -mem 50g -cores 4 -q x86_1h python run/evaluate.py -p=GEX2ATAC.h5ad GEX2ATAC`
