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
from samples_generator import SnippetInferInstance
from z_privacy_attack.metrics import MetricCalculator,F1Calculator,AUCCalculator,TPRFPRCalculator


from typing import List


class MLP(nn.Module):
    """A basic PyTorch model.
    3-Layer Multi-Layer Perceptron with ReLU activation.

    Note that PyTorch models must inherit from nn.Module.
    """
    def __init__(
            self,
            snp_dim: int = 256,
            ctx_dim: int = 64,
            hid_dim: int = 256,
            dropout: float = 0.2) -> None:
        """Constructor function for the MLP model.
        We define the layers of the neural network here.

        Args:
            snp_dim (int, optional): Dimension for snippet embedding.
                Defaults to 768.
            ctx_dim (int, optional): Dimension for code context embedding.
                Defaults to 768.
            hid_dim (int, optional): Size of hidden dimension.
                Defaults to 256.
            dropout (float, optional): Dropout rate.
                Defaults to 0.2.
        """
        super().__init__()
        self.proj = nn.Linear(snp_dim + ctx_dim, hid_dim)
        self.act1 = nn.ReLU()
        self.ffwd = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, 2))

    def forward(
            self,
            snippet: torch.Tensor,
            context: torch.Tensor):
        """Forward pass for the MLP model."""

        # the forward function defines how the input
        # will be processed within the model
        # In this model, we concatenate the snippet and context
        # and then pass the concatenated vector to the hidden layers
        x = torch.cat([snippet, context], dim=1)
        x = self.proj(x)
        x = self.act1(x)
        x = self.ffwd(x)
        return x


class SnippetDataset(Dataset):
    def __init__(self, dataset: List[SnippetInferInstance]):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class SnippetCollator:
    def __init__(self) -> None:
        pass

    def __call__(self, batch: List[SnippetInferInstance]):
        snippet_embds = [snippet.snippet_embd for snippet in batch]
        context_embds = [snippet.context_embd for snippet in batch]
        labels = [snippet.label for snippet in batch]

        return (
            torch.tensor(np.array(snippet_embds)),
            torch.tensor(np.array(context_embds)),
            torch.tensor(labels, dtype=torch.long)
        )


def train(
        e: int,
        train_loader: DataLoader,
        model: nn.Module,
        optimizer: Optimizer,
        loss_func: nn.Module,
        device: torch.device):
    # during training, we perform a forward pass and
    # compute the loss and update the weights

    model.train()

    tot_loss = 0.0
    tot_acc = 0.0
    metrics: List[MetricCalculator] = [F1Calculator()]

    progress = tqdm(train_loader)
    # iterate over the training set
    for bid, (snp_input, ctx_input, label) in enumerate(progress):
        # load input data to the designated device (GPU here)
        snp_input = snp_input.to(device)
        ctx_input = ctx_input.to(device)
        label = label.to(device)

        # feed the input into the model to get model predictions
        outputs = model(snp_input, ctx_input)
        preds = torch.argmax(outputs, dim=1)

        # compute the loss
        loss = loss_func(outputs, label)

        # call the optimizer to perform backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute evaluation metrics and print them
        tot_loss += loss.item()
        tot_acc += (preds == label).float().mean().item()
        for m in metrics:
            m.update(outputs, label)

        avg_loss = tot_loss / (bid + 1)
        avg_acc = tot_acc / (bid + 1)

        progress.set_description(
            f'| E {e:2d} | Loss {avg_loss:.4f} | Acc {avg_acc:.4f} |')

    metric_dict = merge_metric_dict(*[m.summary() for m in metrics])
    metric_str = format_metric_string(metric_dict)
    print(
        f'| Training {e:2d} | Loss {avg_loss:.4f} | Acc {avg_acc:.4f}',
        metric_str)

    return {
        'epoch': e,
        'loss': avg_loss,
        'acc': avg_acc,
        **metric_dict
    }


def evaluate(
        e: int,
        eval_loader: DataLoader,
        model: nn.Module,
        loss_func: nn.Module,
        device: torch.device):
    model.eval()
    tot_loss = 0.0
    tot_acc = 0.0
    n_samples = 0
    metrics: List[MetricCalculator] = [
        F1Calculator(), AUCCalculator(), TPRFPRCalculator()]

    with torch.no_grad():
        for snp_input, ctx_input, label in eval_loader:
            snp_input = snp_input.to(device)
            ctx_input = ctx_input.to(device)
            label = label.to(device)

            outputs = model(snp_input, ctx_input)
            preds = torch.argmax(outputs, dim=1)
            loss = loss_func(outputs, label)

            tot_loss += loss.item() * label.shape[0]
            tot_acc += (preds == label).float().sum().item()
            n_samples += label.shape[0]
            for m in metrics:
                m.update(outputs, label)

        avg_loss = tot_loss / n_samples
        avg_acc = tot_acc / n_samples
        metric_dict = merge_metric_dict(*[m.summary() for m in metrics])

        metric_str = format_metric_string(metric_dict)
        print(
            f'| Evaluation {e:2d} | Loss {avg_loss:.4f} | Acc {avg_acc:.4f}',
            metric_str)

    return {
        'epoch': e,
        'loss': avg_loss,
        'acc': avg_acc,
        **metric_dict
    }


def train_val_split(data: List[SnippetInferInstance], val_size: int = 250):
    val_size //= 2
    val_pos = 0
    val_neg = 0
    train_data = []
    val_data = []
    idx = 0
    while val_pos < val_size or val_neg < val_size:
        if data[idx].label == 1:
            if val_pos < val_size:
                val_data.append(data[idx])
                val_pos += 1
            else:
                train_data.append(data[idx])
        else:
            if val_neg < val_size:
                val_data.append(data[idx])
                val_neg += 1
            else:
                train_data.append(data[idx])
        idx += 1

    for idx in range(idx, len(data)):
        train_data.append(data[idx])

    assert len(val_data) == val_size * 2, len(val_data)
    assert len(train_data) == len(data) - val_size * 2, len(train_data)

    return train_data, val_data


def imbalance_dataset(
        data: List[SnippetInferInstance],
        n_positives: int, n_negatives: int):
    n_pos = 0
    n_neg = 0
    idx = 0
    filtered_data = []
    while n_pos < n_positives or n_neg < n_negatives:
        if idx >= len(data):
            break
        current = data[idx]
        if current.label == 1 and n_pos < n_positives:
            filtered_data.append(current)
            n_pos += 1
        if current.label == 0 and n_neg < n_negatives:
            filtered_data.append(current)
            n_neg += 1
        idx += 1

    assert len(filtered_data) == n_positives + n_negatives
    assert len([x for x in filtered_data if x.label == 1]) == n_positives
    assert len([x for x in filtered_data if x.label == 0]) == n_negatives

    return filtered_data


def setup(dataset_name):
    print(f'\n\n {dataset_name}')
    BATCH_SIZE = 32
    EPOCHS = 20
    SEED = 1337

    torch.manual_seed(SEED)

    data_dir = '/home/liwei/privacy-attack-code-model/lw/learning-program-representation/data/poj104clone/'

    data = pickle.load(
        open(os.path.join(data_dir, dataset_name), 'rb'))
    print(f"data size {len(data)}")
    train_data, val_data = train_val_split(data)
    train_dataset = SnippetDataset(train_data)
    val_dataset = SnippetDataset(val_data)

    print(f'Train size: {len(train_dataset)}')
    print(f'Validation size: {len(val_dataset)}')

    # define dataloaders
    # dataloaders are used for loading samples in batches efficiently
    collator = SnippetCollator()
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        collate_fn=collator, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        collate_fn=collator, shuffle=False)

    # set device to GPU
    device = torch.device('cuda:0')

    # create a model, and deploy the model to the designated device (GPU)
    context_embed_shape = data[0].context_embd.shape[0]
    snippet_embed_shape = data[0].snippet_embd.shape[0]
    print(f'context embedding shape: {context_embed_shape}, snippet embedding shape:{snippet_embed_shape}')
    model = MLP(snp_dim=snippet_embed_shape,ctx_dim=context_embed_shape).to(device)
    print(model)
    # create a loss function and an optimizer
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    loss_func = nn.CrossEntropyLoss()

    # training epochs here
    best_score = 0
    best_perfs = None
    for e in range(EPOCHS):
        train(e, train_loader, model, optimizer, loss_func, device)
        eval_perf = evaluate(e, val_loader, model, loss_func, device)

        # if the model performs better than the best model, save the model
        if eval_perf['auc'] > best_score:
            best_score = eval_perf['auc']
            best_perfs = eval_perf
            torch.save(model.state_dict(), './ckpts/best_model.pt')

    # load and evaluate the best model
    print(f'Loading best model: {best_perfs}')
    model.load_state_dict(torch.load('./ckpts/best_model.pt'))
    eval_perf = evaluate(0, val_loader, model, loss_func, device)
    print(f'Performance on balanced validation set: {eval_perf}')

def main():
    names = ['ModelType.BiLSTM.pl_emb_and_BiLSTM_snippet.pkl',
             'ModelType.TreeLSTM.pl_emb_and_BiLSTM_snippet.pkl',
             'ModelType.GCN.pl_emb_and_BiLSTM_snippet.pkl',
             'ModelType.GGNN.pl_emb_and_BiLSTM_snippet.pkl']
    for name in names:
        setup(name)

if __name__ == '__main__':
    main()
