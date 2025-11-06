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
pip install git+https://github.com/abhinadduri/panspecies-dti.git
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

### Single Lit-PCBA
```
# Setup MMseq2
`conda install -c conda-forge -c bioconda mmseqs2`
# Running Single Lit-PCBA
ultrafast-train --exp-id LitPCBA --config configs/saprot_agg_config.yaml --task merged --epochs 15 --ship-model --model-size large --target-protein-id {TARGET} --similarity-threshold {THRESHOLD} --eval-pcba 
```
Targets ids can be found here: ``targets.txt``.

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
Links to download pre-trained models used for Lit-PCBA evaluation in Table 2 are in [checkpoints/README.md](checkpoints/README.md).

# Embed proteins and molecules
Embed a library of proteins/molecules, using `--data-file`: a CSV/TSV file (separator inferred). The `--data-file` to embed must contain a "SMILES" or "Target Sequence" column for drug or target embedding, respectively.

If using a SaProt trained checkpoint, the "Target Sequence" should be a structure-aware sequence with residue and structure tokens (e.g. "RaTcIqAvKvQqIwQdMfVd"). Structure-aware sequences can be generated following [Generate SaProt sequence for a given protein structure](#generate-saprot-sequence-for-a-given-protein-structure). If no structure tokens are detected, a mask token will be used for each resiude's structure token. 
# 

```
# Get target embeddings with pre-trained model
ultrafast-embed --data-file data/DAVIS/test_foldseek.csv  \
    --checkpoint checkpoints/saprot.ckpt \
    --moltype target \ 
    --output-path results/DAVIS_test_target_embeddings.npy

# Get drug embeddings with pre-trained model
ultrafast-embed --data-file data/DAVIS/test_foldseek.csv  \
    --checkpoint checkpoints/saprot.ckpt \
    --moltype drug \ 
    --output-path results/DAVIS_test_drug_embeddings.npy
```
# Vector Database
The following section details the usage of a [ChromaDB](https://docs.trychroma.com/docs/overview/introduction) for ultrafast retrieval of DTIs. Note that creation of the database is a computationally costly preprocessing step, but it only needs to be done once for a given library.
### Make a vector database of drugs
```
ultrafast-store --data-file data/DAVIS/test_foldseek.csv  \
    --embeddings results/DAVIS_test_drug_embeddings.npy \
    --moltype drug \
    --db_dir ./dbs \
    --db_name davis_test_drug_embeddings
```

### Report top-k accuracy by querying targets against the drug database
```
ultrafast-report --data-file data/DAVIS/test_foldseek.csv  \
    --embeddings results/DAVIS_test_target_embeddings.npy \
    --moltype target \
    --db_dir ./dbs \
    --db_name biosnap_test_drug_embeddings \
    --topk 100
```

# Compute TopK Hits for a given Query
This section details finding the TopK hits without using ChromaDB. This is likely faster if you are only going to query a library a few times or if you can massively parallelize the TopK search.

We can compute the TopK hits for a set of targets against a database of drugs.
```
ultrafast-topk --library-embeddings results/DAVIS_test_drug_embeddings.npy \
    --library-type drug --library-data data/DAVIS/test_foldseek.csv \
    --query-embeddings results/DAVIS_test_target_embeddings.npy \
    --query-data data/DAVIS/test_foldseek.csv \
    -K 100
```
or we can compute the TopK hits for a set of drugs against a database of targets by swapping the library and query arguments and changing the `--library-type`.

If you are computing the TopK hits for a large database, it is often faster to break it up into smaller chunks and compute the TopK per chunk in a parallel fashion. The chunks can be combined at the end to get the final TopK hits for the entire database:
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

# Training from scratch
When training a SPRINT DTI Classification model from scratch, you need train/val/test CSV files with the following columns:
```
SMILES,Target Sequence,Label
```
where `SMILES` is the drug SMILES string, `Target Sequence` is the amino acid sequence of the target, and `Label` is a 0/1 value to indicate non-binding or binding, respectively.

CSV files should be placed in `data/custom/`

Models utilizing SaProt as the `--target-featurizer` must have structure-aware sequences in the `Target Sequence` column and the CSVs should be renamed to `*_foldseek.csv` where `*` is train/val/test.

Models can be trained using:
```
ultrafast-train --exp-id custom --task custom --config configs/saprot_agg_config.yaml --model-size large
```
