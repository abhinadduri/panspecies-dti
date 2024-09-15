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

# Embed proteins and molecules
```
# Get embeddings with pre-trained model
ultrafast-embed --data-file data/BIOSNAP/full_data/train.csv  \
    --checkpoint checkpoints/saprot_agg_contrast_biosnap_maxf1.ckpt \
    --output_path results/embeddings.npy
```

# Check top-k accuracy of the model using test data
TODO

# Predict new drug-target interactions
TODO
