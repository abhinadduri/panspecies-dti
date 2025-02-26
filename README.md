# SPRINT
Code for the paper [Scaling Structure Aware Virtual Screening to Billions of Molecules with SPRINT](https://arxiv.org/abs/2411.15418) and the MLSB 2024 paper [SPRINT: Ultrafast Drug-Target Interaction Prediction with Structure-Aware Protein Embeddings](https://arxiv.org/abs/2411.15418v1).

**S**tructure-aware **PR**otein ligand **INT**eraction (SPRINT) is a ultrafast deep learning framework for drug-target interaction prediction and binding affinity prediction.

SPRINT can be used in a Google Colab notebook here:
[![ColabScreen](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vSYzL_KvzyGHhIDq3h3qcITB8cuxE4ZU?usp=sharing)

All datasets are located in the `data` folder.

## Overview
The protein and ligand are co-embedded in a shared space, enabling interaction prediction at the speed of a single dot product.
Proteins are embedded with [SaProt](https://github.com/westlake-repl/SaProt), followed by a Attention-Pooling layer, and small MLP. Ligands are embedded using Morgan Fingerprints and a small MLP.
The model is trained in a fully supervised manner to predict the interaction between proteins and ligands.

# Install
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

## Download MERGED dataset
Script to download splits and data:
```
cd data/MERGED/huge_data/
bash download.sh
cd -
```


# Reproducing the paper
Reproducing the drug-target interaction models in the paper.
### DTI Prediction
The code below reproduces the DTI prediction on the DAVIS dataset.
```
# Reproducing ConPLex
ultrafast-train --exp-id DAVIS --config configs/conplex_config.yaml
# ConPLex-attn
ultrafast-train --exp-id DAVIS --config configs/saprot_agg_config.yaml --prot_proj agg
# SPRINT-sm
ultrafast-train --exp-id DAVIS --config configs/saprot_agg_config.yaml 
# SPRINT
ultrafast-train --exp-id DAVIS --config configs/saprot_agg_config.yaml --model-size large
```
Other DTI dataset models can be reproduced by adding ``--task`` to the commandline with: ``biosnap``, ``bindingdb``, ``biosnap_prot``(Unseen Targets), ``biosnap_mol``(Unseen Drugs), or ``merged``

### Lit-PCBA
```
# SPRINT
ultrafast-train --exp-id LitPCBA --config configs/saprot_agg_config.yaml --epochs 15 --ship-model data/MERGED/huge_data/uniprots_excluded_at_90.txt
# SPRINT-Average
ultrafast-train --exp-id LitPCBA --config configs/saprot_agg_config.yaml --prot-proj avg --epochs 15 --ship-model data/MERGED/huge_data/uniprots_excluded_at_90.txt 
# SPRINT-ProtBert
ultrafast-train --exp-id LitPCBA --config configs/saprot_agg_config.yaml --target-featurizer ProtBertFeaturizer --epochs 15 --ship-model data/MERGED/huge_data/uniprots_excluded_at_90.txt 
```
Adding ``--eval-pcba`` can show the performance on the Lit-PCBA dataset after epoch of training.

### TDC Leaderboard
```
# SPRINT
ultrafast-train --exp-id TDC --config configs/TDC_config.yaml 
# SPRINT-ProtBert
ultrafast-train --exp-id TDC --config configs/TDC_config.yaml --target-featurizer ProtBertFeaturizer
# SPRINT-ESM2
ultrafast-train --exp-id TDC --config configs/TDC_config.yaml --target-featurizer ESM2Featurizer
```

# Download pre-trained model
Links to download pre-trained models are in `checkpoints/README.md`.

Once downloaded, just `gunzip` the file to get the ready-to-use model checkpoint.

# Embed proteins and molecules
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
# Vector Database
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
