import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from nets import ALBERT
from datasets import DataGenerator
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score


def train_bert(model_name):
    device = torch.device("cuda")
    if model_name == "albert":
        model = ALBERT(num_classes=7).to(device)
    else:
        raise ValueError("model name error")
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    loss_func = nn.CrossEntropyLoss()
    tokenizer = BertTokenizer.from_pretrained("albert-chinese-small")
    train_loader = DataLoader(DataGenerator(root="datasets/data_train.csv", tokenizer=tokenizer, max_len=200),
                              batch_size=16, shuffle=True)
    val_loader = DataLoader(DataGenerator(root="datasets/data_val.csv", tokenizer=tokenizer, max_len=200), batch_size=32,
                            shuffle=False)

    train_losses, train_accs, train_pres, train_recs, train_f1s = [], [], [], [], []
    val_losses, val_accs, val_pres, val_recs, val_f1s = [], [], [], [], []
    max_acc = 0
    for epoch in range(30):
        train_acc, train_loss = train_one_epoch(model, train_loader, loss_func, optimizer, scheduler, device, epoch)
        val_acc, val_loss = get_val_result(model, val_loader, loss_func, device)

        train_accs.append(train_acc)
        train_losses.append(train_loss)

        val_accs.append(val_acc)
        val_losses.append(val_loss)

        if val_acc > max_acc:
            max_acc = val_acc
            torch.save(model.state_dict(), f"models/{model_name}_best.pth")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"models/{model_name}_epoch{epoch + 1}.pth")

        print(f"epoch:{epoch + 1},train_acc:{train_acc:.4f},val_acc:{val_acc:.4f},train_loss:{train_loss:.4f},val_loss:{val_loss:.4f}")

    plot_acc_loss(train_accs, train_losses, val_accs, val_losses, model_name)


def train_one_epoch(model, train_loader, loss_func, optimizer, scheduler, device, epoch):
    data_train = tqdm(train_loader)
    losses = []
    labels_true, labels_pred = np.array([]), np.array([])
    model.train()
    for batch, (x, y, z) in enumerate(data_train):
        labels_true = np.concatenate([labels_true, z.numpy()], axis=-1)
        input_ids, attention_mask, labels = x.to(device), y.to(device), z.to(device)
        y_prob = model(input_ids, attention_mask)
        y_pred = torch.argmax(y_prob, dim=-1).cpu().numpy()
        labels_pred = np.concatenate([labels_pred, y_pred], axis=-1)
        loss = loss_func(y_prob, labels)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        data_train.set_description_str(
            f"epoch:{epoch + 1},batch:{batch + 1},loss:{loss.item():.5f},lr:{scheduler.get_last_lr()[0]:.4f}")
    scheduler.step()

    train_acc = accuracy_score(labels_true, labels_pred)
    train_loss = float(np.mean(losses))

    return train_acc, train_loss


def get_val_result(model, val_loader, loss_func, device):
    data_val = tqdm(val_loader)
    losses = []
    labels_true, labels_pred = np.array([]), np.array([])
    model.eval()
    with torch.no_grad():
        for x, y, z in data_val:
            labels_true = np.concatenate([labels_true, z.numpy()], axis=-1)
            input_ids, attention_mask, labels = x.to(device), y.to(device), z.to(device)
            y_prob = model(input_ids, attention_mask)
            y_pred = torch.argmax(y_prob, dim=-1).cpu().numpy()
            labels_pred = np.concatenate([labels_pred, y_pred], axis=-1)
            loss = loss_func(y_prob, labels)
            losses.append(loss.item())

    val_acc = accuracy_score(labels_true, labels_pred)
    val_loss = float(np.mean(losses))

    return val_acc, val_loss


def plot_acc_loss(train_accs, train_losses, val_accs, val_losses, model_name):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_accs) + 1), train_accs, "r", label="train")
    plt.plot(range(1, len(val_accs) + 1), val_accs, "g", label="val")
    plt.title(f"{model_name}_accuracy-epoch")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_losses) + 1), train_losses, "r", label="train")
    plt.plot(range(1, len(val_losses) + 1), val_losses, "g", label="val")
    plt.title(f"{model_name}_loss-epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(f"images/{model_name}_epoch_acc_loss.jpg")


if __name__ == '__main__':
    train_bert(model_name="albert")
