import os
import sys
import unittest
from ultrafast.train import train_cli, train
from ultrafast.embed import embed_cli, embed

class TestDefaults(unittest.TestCase): 
    def test_train_default(self):
        cmd = 'ultrafast-train --config configs/unittest_config.yaml --exp-id unittest --epochs 1 --no-wandb'
        sys.argv = cmd.split()
        train_cli()
        assert os.path.exists('best_models/temp/davis.ckpt')
    
    def test_embed_default(self):
        cmd = 'ultrafast-train --config configs/unittest_config.yaml --exp-id unittest --epochs 1 --no-wandb'
        sys.argv = cmd.split()
        train_cli()
        cmd = 'ultrafast-embed --data-file data/unittest_dummy_data/train.csv  --checkpoint best_models/unittest/unittest.ckpt --output_path results/embeddings.npy'
        sys.argv = cmd.split()
        embed_cli()
        assert os.path.exists('results/embeddings.npy')


if __name__ == '__main__':
    unittest.main()
