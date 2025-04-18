import torch
import numpy as np
from utils.metric import Confusion
from sklearn.cluster import KMeans


def get_mean_embeddings(bert, input_ids, attention_mask):
    """
    get the mean embeddings of batch
    """
    bert_output = bert.forward(input_ids=input_ids, attention_mask=attention_mask)
    attention_mask = attention_mask.unsqueeze(-1)
    mean_output = torch.sum(bert_output[0] * attention_mask, dim=1) / torch.sum(
        attention_mask, dim=1
    )
    return mean_output


def get_batch_token(tokenizer, text, max_length):
    """
    use the tokenizer of bert to encode the text inputs
    """
    token_feat = tokenizer.batch_encode_plus(
        text,
        max_length=max_length,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
    return token_feat


def get_kmeans_centers(bert, tokenizer, train_loader, num_classes, max_length):
    """
    get the kmeans centers before training
    """
    for i, batch in enumerate(train_loader):
        text, label = batch["text"], batch["label"]
        tokenized_features = get_batch_token(tokenizer, text, max_length)
        for key in tokenized_features.keys():
            tokenized_features[key] = tokenized_features[key].to(bert.device)
        corpus_embeddings = get_mean_embeddings(bert, **tokenized_features).cpu()

        if i == 0:
            all_labels = label
            all_embeddings = corpus_embeddings.detach().numpy()
        else:
            all_labels = torch.cat((all_labels, label), dim=0)
            all_embeddings = np.concatenate(
                (all_embeddings, corpus_embeddings.detach().numpy()), axis=0
            )

    # Perform KMeans clustering
    confusion = Confusion(num_classes)
    clustering_model = KMeans(n_clusters=num_classes)
    clustering_model.fit(all_embeddings)
    cluster_assignment = clustering_model.labels_

    true_labels = all_labels  # convert 1,2,3,4 label to 0,1,2,3
    pred_labels = torch.tensor(cluster_assignment)
    print(
        "all_embeddings:{}, true_labels:{}, pred_labels:{}".format(
            all_embeddings.shape, len(true_labels), len(pred_labels)
        )
    )

    confusion.add(pred_labels, true_labels)
    confusion.optimal_assignment(num_classes)
    print(
        "Iterations:{}, Clustering ACC:{:.3f}, centers:{}".format(
            clustering_model.n_iter_,
            confusion.acc(),
            clustering_model.cluster_centers_.shape,
        )
    )

    return clustering_model.cluster_centers_
