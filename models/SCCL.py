import os
import time
import numpy as np
from tqdm import tqdm
from sklearn import cluster

from utils.utils import statistics_log
from utils.metric import Confusion
from dataloader import unshuffle_loader

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
from utils.cluster_utils import target_distribution
from utils.contrastive_utils import PairConLoss
from models import SCCLFeatureExtractor
from transformers import AutoModel, AutoTokenizer


class SCCL(nn.Module):

    def __init__(
        self,
        model: SCCLFeatureExtractor,
        tokenizer: AutoTokenizer,
        optimizer: torch.optim.AdamW,  # or torch.optim.Adam
        train_loader: DataLoader,
        args,
    ):
        super(SCCL, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.args = args
        self.eta = self.args.eta

        self.cluster_loss = nn.KLDivLoss(size_average=False)
        self.contrast_loss = PairConLoss(temperature=self.args.temperature)

        self.gstep = 0
        print(f"| SCCL config | temp:{self.args.temperature}, eta:{self.args.eta} |")

    def batch_token(self, text: str):
        """
        use bert to get text token
        """
        token_feat = self.tokenizer.batch_encode_plus(
            text,
            max_length=self.args.max_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return token_feat

    def get_bert_input(self, batch: torch.Tensor):
        # if use data augment
        if len(batch) == 4:
            text1, text2, text3 = (
                batch["text"],
                batch["augmentation_1"],
                batch["augmentation_2"],
            )
            feat1 = self.batch_token(text1)
            feat2 = self.batch_token(text2)
            feat3 = self.batch_token(text3)

            input_ids = torch.cat(
                [
                    feat1["input_ids"].unsqueeze(1),
                    feat2["input_ids"].unsqueeze(1),
                    feat3["input_ids"].unsqueeze(1),
                ],
                dim=1,
            )
            attention_mask = torch.cat(
                [
                    feat1["attention_mask"].unsqueeze(1),
                    feat2["attention_mask"].unsqueeze(1),
                    feat3["attention_mask"].unsqueeze(1),
                ],
                dim=1,
            )
        # not use data augment
        elif len(batch) == 2:
            text = batch["text"]
            feat1 = self.batch_token(text)
            feat2 = self.batch_token(text)

            input_ids = torch.cat(
                [feat1["input_ids"].unsqueeze(1), feat2["input_ids"].unsqueeze(1)],
                dim=1,
            )
            attention_mask = torch.cat(
                [
                    feat1["attention_mask"].unsqueeze(1),
                    feat2["attention_mask"].unsqueeze(1),
                ],
                dim=1,
            )

        # input_ids: [B, N,]
        return input_ids.cuda(), attention_mask.cuda()

    def train_step(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        two type train method: virtual and explicit
        """
        # if data aug type is virtual
        if self.args.augtype == "virtual":
            embd1, embd2 = self.model(input_ids, attention_mask, task_type="virtual")
            # Instance-CL loss
            feat1, feat2 = self.model.contrast_logits(embd1, embd2)
        else:  # if data aug type is explicit
            embd1, embd2, embd3 = self.model(
                input_ids, attention_mask, task_type="explicit"
            )
            # Instance-CL loss
            feat1, feat2 = self.model.contrast_logits(embd2, embd3)

        losses = self.contrast_loss(feat1, feat2)
        loss = self.eta * losses["loss"]

        # Clustering loss
        if self.args.objective == "SCCL":
            output = self.model.get_cluster_prob(embd1)
            target = target_distribution(output).detach()

            cluster_loss = (
                self.cluster_loss((output + 1e-08).log(), target) / output.shape[0]
            )
            # different loss compute
            loss += (
                0.5 * cluster_loss if self.args.augtype == "virtual" else cluster_loss
            )
            losses["cluster_loss"] = cluster_loss.item()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses

    def train(self):
        """
        train model
        """
        print(
            f"| Iterations: {self.args.max_iter} | Batches: {len(self.train_loader)} |"
        )

        self.model.train()
        for i in tqdm(np.arange(self.args.max_iter + 1)):
            try:
                batch = next(train_loader_iter)  # shape=[B, N, ] input_id
            except:
                train_loader_iter = iter(self.train_loader)
                batch = next(train_loader_iter)

            input_ids, attention_mask = self.get_bert_input(
                batch
            )  # shape=[B, N, D] embedding_dim

            losses = self.train_step(input_ids, attention_mask)

            if (self.args.print_freq > 0) and (
                (i % self.args.print_freq == 0) or (i == self.args.max_iter)
            ):
                statistics_log(self.args.tensorboard, losses=losses, global_step=i)
                self.evaluate(i)
                self.model.train()

        return None

    def evaluate(self, step: int):
        """
        evaluate data according to the training step
        """
        dataloader = unshuffle_loader(self.args)
        print(f"| Evaluating Batches | DataLoader Len: {len(dataloader)} |")

        self.model.eval()
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                text, label = batch["text"], batch["label"]
                feat = self.batch_token(text)
                embeddings = self.model(
                    feat["input_ids"].cuda(),
                    feat["attention_mask"].cuda(),
                    task_type="evaluate",
                )

                model_prob = self.model.get_cluster_prob(embeddings)
                if i == 0:
                    all_labels = label
                    all_embeddings = embeddings.detach()
                    all_prob = model_prob
                else:
                    all_labels = torch.cat((all_labels, label), dim=0)
                    all_embeddings = torch.cat(
                        (all_embeddings, embeddings.detach()), dim=0
                    )
                    all_prob = torch.cat((all_prob, model_prob), dim=0)

        # Initialize confusion matrices
        confusion, confusion_model = Confusion(self.args.num_classes), Confusion(
            self.args.num_classes
        )

        all_pred = all_prob.max(1)[1]
        confusion_model.add(all_pred, all_labels)
        confusion_model.optimal_assignment(self.args.num_classes)
        acc_model = confusion_model.acc()

        kmeans = cluster.KMeans(
            n_clusters=self.args.num_classes, random_state=self.args.seed
        )
        embeddings = all_embeddings.cpu().numpy()
        kmeans.fit(embeddings)
        pred_labels = torch.tensor(kmeans.labels_.astype(np.int32))

        # clustering accuracy
        confusion.add(pred_labels, all_labels)
        confusion.optimal_assignment(self.args.num_classes)
        acc = confusion.acc()

        ressave = {"acc": acc, "acc_model": acc_model}
        ressave.update(confusion.clusterscores())
        for key, val in ressave.items():
            self.args.tensorboard.add_scalar("Test/{}".format(key), val, step)

        print(f"[Representation] Clustering scores: {confusion.clusterscores()}")
        print(f"[Representation] ACC: {acc:.3f}")
        print(f"[Model] Clustering scores:{confusion_model.clusterscores()}")
        print(f"[Model] ACC: {acc_model:.3f}")
        return None
