# Reproducing Molecular Property Prediction

Reproducing molecular property regression tasks (Delaney, FreeSolv, Lipophilicity)
1. Embed molecules with SPRINT
2. Run `benchmark_mpp_regression.py`

```sh
# Embed molecules
ultrafast-embed --data-file lipo/lipo_labels.csv  \
    --checkpoint ../checkpoints/sprint.ckpt \
    --moltype drug \
    --output-path lipo/lipo_drug_embeddings.npy

# Morgan Fingerprint
python benchmark_mpp_regression.py \
    --embeddings lipo/lipo_drug_embeddings.npy \
    --labels lipo/lipo_labels.csv \
    --embedding_type morgan \
    --model mlp

# SPRINT Embedding
python benchmark_mpp_regression.py \
    --embeddings lipo/lipo_drug_embeddings.npy \
    --labels lipo/lipo_labels.csv \
    --embedding_type conplex \
    --model mlp

# SPRINT Embedding + Morgan Fingerprint
python benchmark_mpp_regression.py \
    --embeddings lipo/lipo_drug_embeddings.npy \
    --labels lipo/lipo_labels.csv \
    --embedding_type combined \
    --model mlp
```

Reproducing molecular property classification tasks (ToxCast, Tox21, SIDER)
1. Embed molecules with SPRINT
2. Run `benchmark_mpp_classification.py`

```sh
# Embed molecules
ultrafast-embed --data-file sider/sider_labels.csv  \
    --checkpoint ../checkpoints/sprint.ckpt \
    --moltype drug \
    --output-path sider/sider_drug_embeddings.npy

# Morgan Fingerprint
python benchmark_mpp_classification.py \
    --embeddings sider/sider_drug_embeddings.npy \
    --labels sider/sider_labels.csv \
    --embedding_type morgan \
    --model mlp

# SPRINT Embedding
python benchmark_mpp_classification.py \
    --embeddings sider/sider_drug_embeddings.npy \
    --labels sider/sider_labels.csv \
    --embedding_type conplex \
    --model mlp

# SPRINT Embedding + Morgan Fingerprint
python benchmark_mpp_classification.py \
    --embeddings sider/sider_drug_embeddings.npy \
    --labels sider/sider_labels.csv \
    --embedding_type combined \
    --model mlp
```