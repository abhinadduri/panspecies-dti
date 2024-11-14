import os
import sys
import unittest
import chromadb
from ultrafast.train import train_cli, train
from ultrafast.embed import embed_cli, embed
from ultrafast.store import store_cli, store
from ultrafast.compute_topk import compute_topk_cli, compute_topk

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
        cmd = 'ultrafast-embed --data-file data/unittest_dummy_data/test.csv --moltype target --checkpoint best_models/unittest/unittest.ckpt --output-path results/prot_embeddings.npy'
        sys.argv = cmd.split()
        embed_cli()
        assert os.path.exists('results/prot_embeddings.npy')

    def test_store_default(self):
        cmd = 'ultrafast-train --config configs/unittest_config.yaml --exp-id unittest --epochs 1 --no-wandb'
        sys.argv = cmd.split()
        train_cli()
        cmd = 'ultrafast-embed --data-file data/unittest_dummy_data/train.csv --moltype drug --checkpoint best_models/unittest/unittest.ckpt --output-path results/drug_embeddings.npy'
        sys.argv = cmd.split()
        embed_cli()
        cmd = 'ultrafast-store --data-file data/unittest_dummy_data/train.csv --embeddings results/drug_embeddings.npy --moltype drug --db_dir ./dbs --db_name unittest_test_drug_embeddings'
        sys.argv = cmd.split()
        store_cli()
        client = chromadb.PersistentClient(path="./dbs")
        collection_names = [col.name for col in client.list_collections()]
        # Check if the collection exists
        assert 'unittest_test_drug_embeddings' in collection_names
        # Check if the collection has the correct number of documents
        collection = client.get_collection(name='unittest_test_drug_embeddings')
        assert collection.count() == 5

    def test_compute_topk(self):
        if not os.path.exists('best_models/unittest/unittest.ckpt'):
            cmd = 'ultrafast-train --config configs/unittest_config.yaml --exp-id unittest --epochs 1 --no-wandb'
            sys.argv = cmd.split()
            train_cli()
        if not os.path.exists('results/prot_embeddings.npy'):
            cmd = 'ultrafast-embed --data-file data/unittest_dummy_data/test.csv --moltype target --checkpoint best_models/unittest/unittest.ckpt --output-path results/prot_embeddings.npy'
            sys.argv = cmd.split()
            embed_cli()
        if not os.path.exists('results/drug_embeddings.npy'):
            cmd = 'ultrafast-embed --data-file data/unittest_dummy_data/train.csv --moltype drug --checkpoint best_models/unittest/unittest.ckpt --output-path results/drug_embeddings.npy'
            sys.argv = cmd.split()
            embed_cli()
        cmd = 'ultrafast-topk --library-data data/unittest_dummy_data/train.csv --library-embeddings results/drug_embeddings.npy --library-type drug --query-data data/unittest_dummy_data/test.csv --query-embeddings results/prot_embeddings.npy -K 3'
        sys.argv = cmd.split()
        compute_topk_cli()



if __name__ == '__main__':
    unittest.main()
