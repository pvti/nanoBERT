from datasets import load_dataset
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader


def tokenize(batch, tokenizer, max_len=64):
    return tokenizer(
        batch["sentence"], padding="max_length", truncation=True, max_length=max_len
    )


def load_data(batch_size=32, max_len=64):
    # Load SST-2 dataset
    dataset = load_dataset("glue", "sst2")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Tokenize the dataset
    dataset = dataset.map(
        lambda batch: tokenize(batch, tokenizer, max_len), batched=True
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Create DataLoaders
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset["validation"], batch_size=batch_size)

    return train_loader, test_loader, tokenizer
