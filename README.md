# Ultra-High-Throughput Virtual Screening

All datasets are located in the `data` folder.

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

# Train a model
Reproducing the drug-target interaction model in the MLCB 2024 abstract.
```
# Default config
ultrafast-train --exp-id mlcb --config configs/default_config.yaml
# Attention pooling
ultrafast-train --exp-id mlcb --config configs/agg_config.yml
```

The example script above will generate ProtBert and store ProtBert per-residue embeddings in a file `data/BIOSNAP/full_data/train.csv.prot.h5`.

The goal to start attention pooling training is to run the above script on all nested `*.csv` files with protein sequences in the data folder.

# Download pre-trained model
Links to download pre-trained models are in `checkpoints/README.md`.

Once downloaded, just `gunzip` the file to get the ready-to-use model checkpoint.

# Download MERGED dataset
Script to download splits and data:
```
cd data/MERGED/huge_data/
bash download.sh
cd -
```

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

# Make a vector database of drugs
```
ultrafast-store --data-file data/BIOSNAP/full_data/test.csv  \
    --embeddings results/BIOSNAP_test_drug_embeddings.npy \
    --moltype drug \
    --db_dir ./dbs \
    --db_name biosnap_test_drug_embeddings
```

# Report top-k accuracy by querying targets against the drug database
```
ultrafast-report --data-file data/BIOSNAP/full_data/test.csv  \
    --embeddings results/BIOSNAP_test_target_embeddings.npy \
    --moltype target \
    --db_dir ./dbs \
    --db_name biosnap_test_drug_embeddings \
    --topk 100
```

# Predict new drug-target interactions
TODO
