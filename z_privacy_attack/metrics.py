import torch

from abc import abstractmethod
from sklearn.metrics import roc_auc_score

from typing import Dict


class MetricCalculator:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def update(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        pass

    @abstractmethod
    def summary(self) -> dict:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class F1Calculator(MetricCalculator):
    def __init__(self) -> None:
        self._tot_hits = 0
        self._tot_trues = 0
        self._tot_preds = 0

    def update(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        preds = outputs.argmax(dim=1)
        self._tot_preds += preds.sum().item()
        self._tot_trues += labels.sum().item()
        hits = torch.logical_and(preds, labels).sum().item()
        self._tot_hits += hits

    def _precision(self) -> float:
        if self._tot_preds == 0:
            return 0.0
        return self._tot_hits / self._tot_preds

    def _recall(self) -> float:
        if self._tot_trues == 0:
            return 0.0
        return self._tot_hits / self._tot_trues

    def _f1(self) -> float:
        if self._tot_preds == 0 and self._tot_trues == 0:
            return 0.0
        return 2 * self._tot_hits / (self._tot_preds + self._tot_trues)

    def summary(self) -> Dict:
        return {
            'precision': self._precision(),
            'recall': self._recall(),
            'f1': self._f1()
        }

    def reset(self) -> None:
        self._tot_hits = 0
        self._tot_trues = 0
        self._tot_preds = 0


class AUCCalculator(MetricCalculator):
    def __init__(self) -> None:
        super().__init__()
        self.label_list = []
        self.pred_list = []

    def reset(self) -> None:
        self.label_list = []
        self.pred_list = []

    def update(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        # outputs: B, n_cls
        # labels: B
        probs = outputs[:, 1]
        self.label_list.extend(labels.cpu().numpy())
        self.pred_list.extend(probs.cpu().numpy())

    def summary(self) -> dict:
        auc = roc_auc_score(self.label_list, self.pred_list)
        return {'auc': auc}


class TPRFPRCalculator(MetricCalculator):
    def __init__(self) -> None:
        super().__init__()
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def update(self, outputs: torch.Tensor, labels: torch.Tensor) -> None:
        preds = outputs.argmax(dim=1)
        self.tp += torch.logical_and(preds, labels).sum().item()
        self.tn += torch.logical_and(torch.logical_not(preds),
                                     torch.logical_not(labels)).sum().item()
        self.fp += torch.logical_and(preds,
                                     torch.logical_not(labels)).sum().item()
        self.fn += torch.logical_and(torch.logical_not(preds),
                                     labels).sum().item()

    def reset(self) -> None:
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def summary(self) -> dict:
        return {
            'tpr': self.tp / (self.tp + self.fn),
            'fpr': self.fp / (self.fp + self.tn),
            # 'tp': self.tp,
            # 'tn': self.tn,
            # 'fp': self.fp,
            # 'fn': self.fn
        }
