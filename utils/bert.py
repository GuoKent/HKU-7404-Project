from transformers import AutoModel, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer
from main import get_args

BERT_CLASS = {
    "distilbert": "distilbert-base-uncased",
}

SBERT_CLASS = {
    "distilbert": "distilbert-base-nli-stsb-mean-tokens",
}


def bert(args):
    if args.use_pretrain == "SBERT":
        bert_model = SentenceTransformer(SBERT_CLASS[args.bert])
        tokenizer = bert_model[0].tokenizer
        model = bert_model[0].auto_model
        print("..... loading Sentence-BERT !!!")
    else:
        config = AutoConfig.from_pretrained(BERT_CLASS[args.bert])
        model = AutoModel.from_pretrained(BERT_CLASS[args.bert], config=config)
        tokenizer = AutoTokenizer.from_pretrained(BERT_CLASS[args.bert])
        print("..... loading plain BERT !!!")

    return model, tokenizer


if __name__ == "__main__":
    args = get_args()
    bert(args)
