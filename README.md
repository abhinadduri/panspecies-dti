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

# Embed proteins and molecules
```
# Get embeddings with pre-trained model
ultrafast-embed --data-file data/BIOSNAP/full_data/train.csv --checkpoint <your_checkpoint>.ckpt --output_path results/embeddings.npy
```
