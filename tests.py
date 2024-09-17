import os
import sys
import unittest
import chromadb
from ultrafast.train import train_cli, train
from ultrafast.embed import embed_cli, embed
from ultrafast.store import store_cli, store

class TestDefaults(unittest.TestCase): 
    def test_train_default(self):
        cmd = 'ultrafast-train --config configs/unittest_config.yaml --exp-id unittest --epochs 1 --no-wandb'
        sys.argv = cmd.split()
        train_cli()
        assert os.path.exists('best_models/unittest/unittest.ckpt')
    
    def test_embed_default(self):
        cmd = 'ultrafast-train --config configs/unittest_config.yaml --exp-id unittest --epochs 1 --no-wandb'
        sys.argv = cmd.split()
        train_cli()
        cmd = 'ultrafast-embed --data-file data/unittest_dummy_data/train.csv  --checkpoint best_models/unittest/unittest.ckpt --output_path results/embeddings.npy'
        sys.argv = cmd.split()
        embed_cli()
        assert os.path.exists('results/embeddings.npy')

    def test_store_default(self):
        cmd = 'ultrafast-train --config configs/unittest_config.yaml --exp-id unittest --epochs 1 --no-wandb'
        sys.argv = cmd.split()
        train_cli()
        cmd = 'ultrafast-embed --data-file data/unittest_dummy_data/train.csv  --checkpoint best_models/unittest/unittest.ckpt --output_path results/embeddings.npy'
        sys.argv = cmd.split()
        embed_cli()
        cmd = 'ultrafast-store --data-file data/unittest_dummy_data/train.csv --embeddings results/embeddings.npy --moltype drug --db_dir ./dbs --db_name unittest_test_drug_embeddings'
        sys.argv = cmd.split()
        store_cli()
        client = chromadb.PersistentClient(path="./dbs")
        collection_names = [col.name for col in client.list_collections()]
        # Check if the collection exists
        assert 'unittest_test_drug_embeddings' in collection_names
        # Check if the collection has the correct number of documents
        collection = client.get_collection(name='unittest_test_drug_embeddings')
        assert collection.count() == 5


if __name__ == '__main__':
    unittest.main()
