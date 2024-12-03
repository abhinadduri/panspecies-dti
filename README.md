# SPRINT
**S**tructure-aware **PR**otein ligand **INT**eraction (SPRINT) is a ultrafast deep learning framework for drug-target interaction prediction. Code for the MLSB 2024 paper [SPRINT: Ultrafast Drug-Target Interaction Prediction with Structure-Aware Protein Embeddings](https://arxiv.org/abs/2411.15418).

All datasets are located in the `data` folder.

<!-- I cannot get this TOC to work. -->
<!-- <details open> -->
<!-- <summary><b>Table of contents</b></summary> -->
<!--  * [Overview](#overview) -->
<!--  * [Install](#install) -->
<!--  * [Train a model](#train-a-model) -->
<!--  * [Model Checkpoints](#download-pre-trained-model) -->
<!--  * [MERGED Dataset](#download-MERGED-dataset) -->
<!--  * [Embed Proteins and Molecules](#embed-proteins-and-molecules) -->
<!--  * [Vector Database](#vector-database) -->
<!--    - [Make a vector database of drugs](#make-a-vector-database-of-drugs) -->
<!--    - [Report top-k accuracy by querying targets against the drug database](#report-top-k-accuracy-by-querying-targets-against-the-drug-database) -->
<!--  * [Compute TopK Hits](#compute-TopK-Hits-for-a-given-Query) -->
<!--  * [Generate SaProt sequence](#generate-SaProt-sequence-for-a-given-protein-structure) -->
<!-- </details> -->

## Overview
The protein and ligand are co-embedded in a shared space, enabling interaction prediction at the speed of a single dot product.
Proteins are embedded with [SaProt](https://github.com/westlake-repl/SaProt), followed by a Attention-Pooling layer, and small MLP. Ligands are embedded using Morgan Fingerprints and a small MLP.
The model is trained in a fully supervised manner to predict the interaction between proteins and ligands.


## Install
```
# Install from source
git clone https://github.com/abhinadduri/panspecies-dti.git
cd panspecies-dti
pip install -e .

# Or install directly from pip
install git+https://github.com/abhinadduri/panspecies-dti.git
```
If you want to use DDP for faster training, first follow the above installation instructions.
Then manually downgrade lightning to 2.0.8 via `pip install lightning==2.0.8`

## Train a model
Reproducing the drug-target interaction model in the MLCB 2024 abstract.
```
# Default config
ultrafast-train --exp-id mlcb --config configs/default_config.yaml
# Attention pooling
ultrafast-train --exp-id mlcb --config configs/agg_config.yml
```

The example script above will generate ProtBert and store ProtBert per-residue embeddings in a file `data/BIOSNAP/full_data/train.csv.prot.h5`.

The goal to start attention pooling training is to run the above script on all nested `*.csv` files with protein sequences in the data folder.

## Download pre-trained model
Links to download pre-trained models are in `checkpoints/README.md`.

Once downloaded, just `gunzip` the file to get the ready-to-use model checkpoint.

## Download MERGED dataset
Script to download splits and data:
```
cd data/MERGED/huge_data/
bash download.sh
cd -
```

## Embed proteins and molecules
```
# Get target embeddings with pre-trained model
ultrafast-embed --data-file data/BIOSNAP/full_data/test.csv  \
    --checkpoint checkpoints/saprot_agg_contrast_biosnap_maxf1.ckpt \
    --moltype target \ 
    --output_path results/BIOSNAP_test_target_embeddings.npy

# Get drug embeddings with pre-trained model
ultrafast-embed --data-file data/BIOSNAP/full_data/test.csv  \
    --checkpoint checkpoints/saprot_agg_contrast_biosnap_maxf1.ckpt \
    --moltype drug \ 
    --output_path results/BIOSNAP_test_drug_embeddings.npy
```
## Vector Database
### Make a vector database of drugs
```
ultrafast-store --data-file data/BIOSNAP/full_data/test.csv  \
    --embeddings results/BIOSNAP_test_drug_embeddings.npy \
    --moltype drug \
    --db_dir ./dbs \
    --db_name biosnap_test_drug_embeddings
```

### Report top-k accuracy by querying targets against the drug database
```
ultrafast-report --data-file data/BIOSNAP/full_data/test.csv  \
    --embeddings results/BIOSNAP_test_target_embeddings.npy \
    --moltype target \
    --db_dir ./dbs \
    --db_name biosnap_test_drug_embeddings \
    --topk 100
```

# Compute TopK Hits for a given Query
We can compute the TopK hits for a set of targets against a database of drugs.
```
ultrafast-topk --library-embeddings results/BIOSNAP_test_drug_embeddings.npy \
    --library-type drug --library-data data/BIOSNAP/full_data/test.csv \
    --query-embeddings results/BIOSNAP_test_target_embeddings.npy \
    --query-data data/BIOSNAP/full_data/test.csv \
    -K 100
```
or we can compute the TopK hits for a set of drugs against a database of targets by swapping the library and query arguments and changing the `--library-type`.

If you are computing the TopK hits for a large database, it is often faster to break it up into smaller chunks and compute the TopK per chunk. The chunks can be combined at the end to get the final TopK hits for the entire database:
```
python utils/combine_chunks.py [directory containing the TopK per chunk] -K 100 -O [output file]
```

# Generate SaProt sequence for a given protein structure
[foldseek](https://github.com/steineggerlab/foldseek) must be installed somewhere in your path.

The protein structure can be in PDB or mmCIF format. The script will generate the sequence of the protein structure and save it to the specified csv file, appending the output to any existing data.
``` 
python utils/structure_to_saprot.py -I [path to the protein structure] --chain [chain of protein] -O [path to the output file]
```
If the protein was **NOT** generated by AF2 or another tool that outputs a confidence score, add `--no-plddt-mask` to the command.

# Predict new drug-target interactions
TODO
