import argparse

def parser():
    p = argparse.ArgumentParser(description="Framework for Mathematical Formula Recognition")
    p.add_argument("--task", choices=['train', 'test'], help="train or test a model", nargs='+')
    p.add_argument("--config", "-c", default="config.yaml", help="config file to use")
    p.add_argument("--resume-from", "-r", default=None, help="resume from file")
    p.add_argument("--test-set", "-t", default=None, help="test_set")
    p.add_argument("--image-path", "-i", default=None, help="image path")
    p.add_argument("--turn-off-wandb", help="turn off wandb", action='store_true')
    p.add_argument("--num-workers", help="change number of workers", type=int)
    return p.parse_args()