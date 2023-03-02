# Leveraging Protein Language Models (PLMs) for encoding Amino Acid Sequences in Drug Target Interaction (DTI)

This repository is a Python and PyTorch implementation for the project which deals with leveraging Protein Language Models for encoding Protein Sequences in Drug Target Interaction. 

A research paper is written based on the results of this work. Tested on the DAVIS/KIBA data set, the code in this repository can achieve the results of the original [GraphDTA](https://academic.oup.com/bioinformatics/article/37/8/1140/5942970) paper (with the CNN module) and all of the Protein Language Models (SeqVec/DistilProtBERT/ProtBERT).

## Requirements
* PyTorch
* dgl
* rdkit
* dgllife
* pytdc
* transformers
* numpy
* pandas
* tqdm
* scikit_learn

## Usage
1. First clone this repository 
`git clone https://code.solutions-lab.ml.aws.dev/brandry/drug-discovery-research.git`
2. Create new conda environments `dti` and `seqvec` using `dti_env.yml` and `seqvec_env.yml` respectively. Commands:
    1. `conda env create -n dti --file requirements/dti_env.yml`
    2. `conda env create -n dti --file requirements/seqvec_env.yml`
3. Activate `seqvec` conda environment using `conda activate seqvec`
4. Download the *SeqVec* weights from the [SeqVec repository](https://github.com/rostlab/SeqVec). Alternativley, this [link](https://rostlab.org/~deepppi/seqvec.zip) can we used for downloading the weights directly. Place the unzipped folder in the `data_processing` folder.
5. Extract the protein embeddings for *SeqVec* (run for DAVIS and KIBA separately) using `python PLM/extract_seqvec_embeddings.py`
6. Activate `dti` conda environment using `conda activate dti`
7. Extract the protein embeddings for *ProtBERT* and *DistilProtBERT* models (run for DAVIS and KIBA separately) using `python PLM/extract_bert_embeddings.py`
8. To train the dti model use `dti_train.py` using the parameters in the argparse. E.g.
    `python dti_train.py --dataset_choice 0 --prot_lm_model_choice 0 --epochs 1500 --batch_size 32`
9. For full description of all the paramters please check `dti_train.py`
10. Then run `python dti_inference.py` using the appropriate parameters to create the pickle files for further processing to create analysis and plotting. E.g. `dti_train.py --dataset_choice 0 --prot_lm_model_choice 0`
11. Finally run `dti_plots.py` to obtain the plots. (Please run the all the PLMs before running the step as the plots are created using all the models.)

## References
[1] [Nguyen, T., Le, H., Quinn, T. P., Nguyen, T., Le, T. D., & Venkatesh, S. (2021). GraphDTA: Predicting drug–target binding affinity with graph neural networks. Bioinformatics, 37(8), 1140-1147.](https://academic.oup.com/bioinformatics/article/37/8/1140/5942970)