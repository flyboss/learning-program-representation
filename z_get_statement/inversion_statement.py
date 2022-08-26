import os
import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from z_privacy_attack.utils import merge_metric_dict, format_metric_string

from tqdm import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from typing import List


class MLP(nn.Module):
    """A basic PyTorch model.
    3-Layer Multi-Layer Perceptron with ReLU activation.

    Note that PyTorch models must inherit from nn.Module.
    """

    def __init__(
            self,
            emb_dim: int,
            hidden_dim: int,
            output_dim: int,
            dropout: float = 0.2) -> None:
        """Constructor function for the MLP model.
        We define the layers of the neural network here.

        Args:
            snp_dim (int, optional): Dimension for snippet embedding.
                Defaults to 768.
            dropout (float, optional): Dropout rate.
                Defaults to 0.2.
        """
        super().__init__()
        self.proj = nn.Linear(emb_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.ffwd = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim))

    def forward(
            self,
            lstm_emb: torch.Tensor):
        x = self.proj(lstm_emb)
        x = self.act1(x)
        x = self.ffwd(x)
        return x


def multilabel_categorical_crossentropy(y_pred, y_true):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
         本文。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat((y_pred_neg, zeros), dim=-1)
    y_pred_pos = torch.cat((y_pred_pos, zeros), dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    loss = neg_loss + pos_loss
    loss = torch.mean(loss)
    return loss


class StatementDataset(Dataset):
    def __init__(self, dataset: List):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class StatementCollator:
    def __init__(self) -> None:
        pass

    def __call__(self, batch: List):
        LSTM_emb = [obj['LSTM_emb'] for obj in batch]
        labels = [obj['label'] for obj in batch]

        return (
            torch.tensor(np.array(LSTM_emb)),
            torch.tensor(labels, dtype=torch.float)
        )

class FiveStatementCollator:
    def __init__(self) -> None:
        pass

    def __call__(self, batch: List):
        LSTM_emb = [obj['LSTM_emb'] for obj in batch]
        labels = [obj['label'][500:505] for obj in batch]

        return (
            torch.tensor(np.array(LSTM_emb)),
            torch.tensor(labels, dtype=torch.float)
        )


class MultiLabelMetrics:
    # https://mmuratarat.github.io/2020-01-25/multilabel_classification_metrics
    def __init__(self):
        pass

    @staticmethod
    def accuracy(y_true: np.array, y_pred: np.array):
        temp = 0
        for i in range(y_true.shape[0]):
            temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
        return temp / y_true.shape[0]


def train(
        e: int,
        train_loader: DataLoader,
        model: nn.Module,
        optimizer: Optimizer,
        loss_func,
        device: torch.device):
    # during training, we perform a forward pass and
    # compute the loss and update the weights

    model.train()

    tot_loss = 0.0
    tot_acc = 0.0

    progress = tqdm(train_loader)
    # iterate over the training set
    for bid, (lstm_emb, label) in enumerate(progress):
        lstm_emb = lstm_emb.to(device)
        label = label.to(device)

        outputs = model(lstm_emb)
        loss = loss_func(outputs, label)

        # call the optimizer to perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute evaluation metrics and print them
        tot_loss += loss.item()
        tot_acc += MultiLabelMetrics.accuracy(label.detach().cpu().numpy(), outputs.detach().cpu().numpy())

        avg_loss = tot_loss / (bid + 1)
        avg_acc = tot_acc / (bid + 1)

        progress.set_description(
            f'| E {e:2d} | Loss {avg_loss:.4f} | Acc {avg_acc:.4f} |')


def evaluate(
        e: int,
        eval_loader: DataLoader,
        model: nn.Module,
        loss_func,
        device: torch.device):
    model.eval()
    tot_loss = 0.0
    tot_acc = 0.0
    n_samples = 0

    with torch.no_grad():
        for lstm_emb, label in eval_loader:
            lstm_emb = lstm_emb.to(device)
            label = label.to(device)

            outputs = model(lstm_emb)
            loss = loss_func(outputs, label)

            tot_loss += loss.item() * label.shape[0]
            tot_acc += MultiLabelMetrics.accuracy(label.detach().cpu().numpy(), outputs.detach().cpu().numpy())
            n_samples += label.shape[0]

        avg_loss = tot_loss / n_samples
        avg_acc = tot_acc / n_samples

        print(
            f'| Evaluation {e:2d} | Loss {avg_loss:.4f} | Acc {avg_acc:.4f}')

    return {
        'epoch': e,
        'loss': avg_loss,
        'acc': avg_acc,
    }


def setup():
    BATCH_SIZE = 16
    EPOCHS = 20
    SEED = 1337
    LEARNING_RATE = 1e-6
    torch.manual_seed(SEED)
    device = torch.device('cuda:3')

    data_dir = '/home/liwei/privacy-attack-code-model/lw/learning-program-representation/data/poj104statement/'

    with open(file=os.path.join(data_dir, 'lstm-train.pkl'), mode='rb') as f:
        train_data = pickle.load(f)
        train_data = train_data[:int(len(train_data) * 0.5)]
    with open(file=os.path.join(data_dir, 'lstm-val.pkl'), mode='rb') as f:
        val_data = pickle.load(f)

    train_dataset = StatementDataset(train_data)
    val_dataset = StatementDataset(val_data)

    print(f'Train size: {len(train_dataset)}')
    print(f'Validation size: {len(val_dataset)}')

    # collator = StatementCollator()
    collator = FiveStatementCollator()


    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        collate_fn=collator, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        collate_fn=collator, shuffle=False)

    # set device to GPU

    lstm_emb_dim = len(train_data[0]['LSTM_emb'])
    output_dim = len(train_data[0]['label'])
    model = MLP(emb_dim=lstm_emb_dim, hidden_dim=64, output_dim=5).to(device)
    print(model)
    # create a loss function and an optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_func = nn.BCEWithLogitsLoss()
    # loss_func = multilabel_categorical_crossentropy

    # training epochs here
    best_score = 0
    best_perfs = None
    for e in range(EPOCHS):
        train(e, train_loader, model, optimizer, loss_func, device)
        eval_perf = evaluate(e, val_loader, model, loss_func, device)

        # if the model performs better than the best model, save the model
        if eval_perf['acc'] > best_score:
            best_score = eval_perf['acc']
            best_perfs = eval_perf
        #     torch.save(model.state_dict(), './ckpts/best_model.pt')

    # load and evaluate the best model
    # print(f'Loading best model: {best_perfs}')
    # model.load_state_dict(torch.load('./ckpts/best_model.pt'))
    # eval_perf = evaluate(0, val_loader, model, loss_func, device)
    # print(f'Performance on balanced validation set: {eval_perf}')


if __name__ == '__main__':
    setup()
    # input = torch.randn(3,3)
    # target = torch.tensor([[0, 1, 1],
    #                        [1, 0, 1],
    #                        [1, 0, 1],
    #                        ])
    # loss = multilabel_categorical_crossentropy(input,target)
    # print(loss)
