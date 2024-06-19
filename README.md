# Benchmarking ConPLex on Bioactivity Datasets
To benchmark ConPLex embeddings on bioactivity task finetuning, we assume that you have access to the conplex-dti script to generate embeddings.

To start, you need a TSV file with molecule ID and molecule SMILES as the two columns. For an example, see the `benchmark_data/moltoxpred_id.tsv` file. We need to generate ConPLex embeddings for this. First, activate the conda environment with the conplex-dti package installed. On my machine:

```
conda activate conplex-screen
```

Then, run the following command to generate embeddings:

```
conplex-dti embed --moltype molecule --data-file benchmark_data/moltoxpred_id.tsv --model-path ../ConPLex-screen/models/ConPLex_v1_BindingDB.pt --outfile benchmark_data/moltoxpred_conplex_embeddings.npz
```

You will need to replace `--model-path` with the path to the ConPLex model you want to use. To run the benchmarking script, you need to also specificy classification labels. For example, with the MolToxPred dataset, you need a TSV file with ID, SMILES, and label columns (see `benchmark_data/moltoxpred_data_processed.tsv`). To run the benchmarking script, run the following command:

```
python benchmark_toxicity.py
```


# MultiModal DTI 

All datasets are located in the `data` folder.

# Training a DTI model (w/o contrastive)

Reproducing the drug-target interaction model in the MLCB 2024 abstract.

First, follow create a `conplex-dti` environment following the instructions here: [ConPLex-screen Compile from Source](https://github.com/cnellington/ConPLex-screen?tab=readme-ov-file#compile-from-source)

Baseline "ConPLex" model:
```
conda activate conplex-dti
python train.py --exp-id mlcb --config default_config.yaml
```

Attention Pooling:
```
conda activate conplex-dti
python train.py --exp-id mlcb --config agg_config.yml
```

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
