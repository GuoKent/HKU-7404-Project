import os
import sys
import torch
import argparse
from models.SCCLFeatureExtractor import SCCLFeatureExtractor
from dataloader import *
from models.SCCL import SCCL
from utils.kmeans import get_kmeans_centers
from utils.utils import setup_path, set_seeds, get_optimizer
from utils.bert import bert
import numpy as np


def main(args):
    args.resPath, args.tensorboard = setup_path(args)
    set_seeds(args.seed)

    # dataset loader
    if args.augtype == "explicit":
        train_loader = explict_augmentation_loader(args)
    else:
        virtual_augmentation_loader(args)

    # model
    torch.cuda.set_device(args.device)
    feature_extractor, tokenizer = bert(args)

    # initialize cluster centers, use k-means
    cluster_centers = get_kmeans_centers(
        feature_extractor, tokenizer, train_loader, args.num_classes, args.max_length
    )

    model = SCCLFeatureExtractor(
        feature_extractor, tokenizer, cluster_centers=cluster_centers, alpha=args.alpha
    )
    model = model.cuda()

    # optimizer
    optimizer = get_optimizer(model, args)

    trainer = SCCL(model, tokenizer, optimizer, train_loader, args)
    trainer.train()

    return None


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="The name of experiment")
    parser.add_argument("--train_instance", type=str, default="local")
    parser.add_argument("--seed", type=int, default=0, help="")
    parser.add_argument("--print_freq", type=float, default=100, help="")
    parser.add_argument("--resdir", type=str, default="./results/")
    parser.add_argument("--s3_resdir", type=str, default="./results")

    parser.add_argument("--bert", type=str, default="distilbert", help="type of bert")
    parser.add_argument(
        "--use_pretrain",
        type=str,
        default="SBERT",
        choices=["BERT", "SBERT", "PAIRSUPCON"],
    )

    # Dataset
    parser.add_argument("--datapath", type=str, default="datasets/")
    parser.add_argument(
        "--dataname", type=str, default="Agnews_charswap_20", help="file name of data"
    )
    parser.add_argument(
        "--num_classes", type=int, default=4, help="the number of clustering"
    )
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--label", type=str, default="label")
    parser.add_argument("--text", type=str, default="text")
    parser.add_argument("--augmentation_1", type=str, default="text1")
    parser.add_argument("--augmentation_2", type=str, default="text2")

    # Learning parameters
    parser.add_argument("--lr", type=float, default=1e-5, help="")
    parser.add_argument("--lr_scale", type=int, default=100, help="")
    parser.add_argument("--max_iter", type=int, default=1000)

    # contrastive learning
    parser.add_argument("--objective", type=str, default="contrastive")
    parser.add_argument(
        "--augtype", type=str, default="virtual", choices=["virtual", "explicit"]
    )
    parser.add_argument("--batch_size", type=int, default=400)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="temperature required by contrastive loss",
    )
    parser.add_argument("--eta", type=float, default=1, help="")

    # Clustering
    parser.add_argument("--alpha", type=float, default=1.0)

    # Device
    parser.add_argument("--device", type=int, default=0, help="rank of gpu")

    args = parser.parse_args(argv)
    args.use_gpu = True if torch.cuda.is_available() else False
    args.resPath = None
    args.tensorboard = None

    return args


if __name__ == "__main__":
    import subprocess

    args = get_args(sys.argv[1:])

    main(args)
