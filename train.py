import os
import copy
import argparse
import torch
import torch.optim as optim
import wandb
from bert import BERTClassifier
from data import load_data

torch.manual_seed(0)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="Hidden layer size"
    )
    parser.add_argument(
        "--num_layers", type=int, default=8, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--ffn_dim", type=int, default=512, help="Feedforward network dimension"
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--ckpt", type=str, default="ckpt", help="Path to save model")
    return parser


def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss, correct = 0, 0
    for batch_idx, batch in enumerate(train_loader):
        input_ids, attention_mask, labels = (
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / len(train_loader.dataset)

    return avg_loss, accuracy


def evaluate(model, test_loader, loss_fn, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["label"].to(device),
            )
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)

    return avg_loss, accuracy


def main():
    parser = get_parser()
    args = parser.parse_args()

    name = f"{args.num_layers}-{args.ffn_dim}-{args.dropout}"
    wandb.init(
        project="nanoBERT-SST2",
        name=name,
        config=vars(args),
    )

    train_loader, test_loader, tokenizer = load_data(batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTClassifier(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ffn_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    acc_best = 0
    model_best = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}:")
        loss_train, acc_train = train(model, train_loader, optimizer, loss_fn, device)
        print(f"Train: Loss = {loss_train:.4f}, Accuracy = {acc_train:.4f}")
        loss_test, acc_test = evaluate(model, test_loader, loss_fn, device)
        print(f"Test: Loss = {loss_test:.4f}, Accuracy = {acc_test:.4f}")

        if acc_test > acc_best:
            acc_best = acc_test
            model_best = copy.deepcopy(model)

        wandb.log(
            {
                "loss_train": loss_train,
                "acc_train": acc_train,
                "loss_test": loss_test,
                "acc_test": acc_test,
                "acc_best": acc_best,
            }
        )

    os.makedirs(args.ckpt, exist_ok=True)
    save_path = os.path.join(args.ckpt, f"{name}-{acc_best:.4f}.pth")
    torch.save(model_best.state_dict(), save_path)
    wandb.finish()


if __name__ == "__main__":
    main()
