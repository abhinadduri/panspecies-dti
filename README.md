# MultiModal DTI 

All datasets are located in the `data` folder.

# Generating per-residue ProtBert embeddings

The new method will need to compute and store per-residue ProtBert embeddings to train an attention pooling layer.

To do this, first make sure you have the conda environment setup from the ConPLeX screen repo: https://github.com/cnellington/ConPLex-screen

Then, use the embed script to take an input file and embed the protein sequences in it:
```
conda activate conplex-screen # or whatever the environment is called on your machine
python embed.py --data-file data/BIOSNAP/full_data/train.csv
```

The example script above will generate ProtBert and store ProtBert per-residue embeddings in a file `data/BIOSNAP/full_data/train.csv.prot.h5`.

The goal to start attention pooling training is to run the above script on all nested `*.csv` files with protein sequences in the data folder.

# Installations

To run these scripts create a new conda environment, and run `conda install rdkit`.
