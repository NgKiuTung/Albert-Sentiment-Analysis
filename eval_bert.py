import torch
import seaborn
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from nets import ALBERT
from datasets import DataGenerator
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import confusion_matrix, precision_recall_curve, accuracy_score, precision_score, f1_score, \
    recall_score, auc, roc_curve

rc = {'font.sans-serif': 'SimHei',
      'axes.unicode_minus': False}
seaborn.set(context='notebook', style='ticks', rc=rc)


def softmax(y):
    y_exp = np.exp(y)
    for i in range(len(y)):
        y_exp[i, :] = y_exp[i, :] / np.sum(y_exp[i, :])

    return y_exp


def one_hot(y, num_classes=2):
    y_ = np.zeros((len(y), num_classes))
    for i in range(len(y)):
        y_[i, int(y[i])] = 1

    return y_


def get_bert_test_result(model_path, columns, model_name, num_classes=2):
    print(model_name)
    device = torch.device("cuda")
    model = ALBERT(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained("albert-chinese-small")
    test_loader = DataLoader(DataGenerator(root="datasets/data_val.csv", tokenizer=tokenizer, max_len=200), batch_size=32,
                             shuffle=False)
    data = tqdm(test_loader)
    labels_true, labels_pred, labels_prob = np.array([]), np.array([]), []
    with torch.no_grad():
        for x, y, z in data:
            datasets_test, mask = x.to(device), y.to(device)
            prob = model(datasets_test, mask)
            labels_prob.append(prob.cpu().numpy())
            pred = torch.argmax(prob, dim=-1).cpu().numpy()
            labels_pred = np.concatenate([labels_pred, pred], axis=-1)
            labels_true = np.concatenate([labels_true, z.numpy()], axis=-1)
    labels_onehot = one_hot(labels_true, num_classes=7)
    labels_prob = nn.Softmax(dim=-1)(torch.FloatTensor(np.concatenate(labels_prob, axis=0))).numpy()

    accuracy = accuracy_score(labels_true, labels_pred)
    precision = precision_score(labels_true, labels_pred, average="macro")
    recall = recall_score(labels_true, labels_pred, average="macro")
    f1 = f1_score(labels_true, labels_pred, average="macro")
    print(f"{model_name},accuracy:{accuracy:.4f},precision:{precision:.4f},recall:{recall:.4f},f1:{f1:.4f}")

    plt.figure(figsize=(10, 10), dpi=300)
    plt.plot([0, 1], [0, 1], "r--")
    fpr, tpr, _ = roc_curve(labels_onehot.ravel(), labels_prob.ravel())
    plt.plot(fpr, tpr, "g", label=f"AUC:{auc(fpr, tpr):.3f}", linewidth=3.0)
    plt.legend(prop={'family': 'Times New Roman', 'size': 10})
    plt.xlabel("FPR", fontsize=20, family='Times New Roman', fontweight='bold', labelpad=22)
    plt.ylabel("TPR", fontsize=20, family='Times New Roman', fontweight='bold', labelpad=25)
    plt.xticks(weight='bold', fontproperties='Times New Roman')
    plt.yticks(weight='bold', fontproperties='Times New Roman')
    plt.tick_params(pad=1.5)
    plt.tick_params(labelsize=20, width=1)
    plt.tight_layout()
    plt.savefig(f"images/{model_name}_roc_curve.jpg", dpi=300)

    plt.figure(figsize=(10, 10), dpi=300)
    p, r, _ = precision_recall_curve(labels_onehot.ravel(), labels_prob.ravel())
    plt.plot(p, r, "g", linewidth=3.0)
    plt.xlabel("Precision", fontsize=20, family='Times New Roman', fontweight='bold', labelpad=22)
    plt.ylabel("Recall", fontsize=20, family='Times New Roman', fontweight='bold', labelpad=25)
    plt.xticks(weight='bold', fontproperties='Times New Roman')
    plt.yticks(weight='bold', fontproperties='Times New Roman')
    plt.tick_params(pad=1.5)
    plt.tick_params(labelsize=20, width=1)
    plt.tight_layout()
    plt.savefig(f"images/{model_name}_pr_curve.jpg", dpi=300)

    matrix = pd.DataFrame(confusion_matrix(labels_true, labels_pred, normalize="true"), columns=columns, index=columns)
    plt.figure(figsize=(10, 10), dpi=300)
    seaborn.heatmap(matrix, annot=True, cmap="GnBu")
    plt.title("confusion_matrix")
    plt.savefig(f"images/{model_name}_confusion_matrix.jpg", dpi=300)


if __name__ == '__main__':
    get_bert_test_result(model_name="albert", model_path="models/albert_best.pth", columns=['like', 'sadness', 'fear', 'anger', 'disgust', 'happiness', 'surprise'], num_classes=7)
