import os
from tensorboardX import SummaryWriter
import random
import torch
import numpy as np


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def setup_path(args):
    resPath = f"exp{args.exp}"
    resPath = args.resdir + resPath
    print(f"results path: {resPath}")

    tensorboard = SummaryWriter(resPath)
    return resPath, tensorboard


def statistics_log(tensorboard, losses=None, global_step=0):
    print("[{}]-----".format(global_step))
    if losses is not None:
        for key, val in losses.items():
            if key in ["pos", "neg", "pos_diag", "pos_rand", "neg_offdiag"]:
                tensorboard.add_histogram("train/" + key, val, global_step)
            else:
                try:
                    tensorboard.add_scalar("train/" + key, val.item(), global_step)
                except:
                    tensorboard.add_scalar("train/" + key, val, global_step)
                print("{}:\t {:.3f}".format(key, val))


def get_optimizer(model, args):

    optimizer = torch.optim.AdamW(
        [
            {"params": model.bert.parameters()},
            {"params": model.contrast_head.parameters(), "lr": args.lr * args.lr_scale},
            {"params": model.cluster_centers, "lr": args.lr * args.lr_scale},
        ],
        lr=args.lr,
    )
    print(optimizer)
    return optimizer
