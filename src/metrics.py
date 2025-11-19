import torch
from typing import Optional, List
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
)


class AccuracyAtK:
    def __init__(self, top_k=1, reduce=None):
        self.top_k = top_k
        self.reduce = reduce

    def __call__(self, logits, labels):
        pred = logits.topk(self.top_k, -1, True, False).indices
        correct = (pred == labels.unsqueeze(-1)).any(1).float()
        if self.reduce == "sum":
            return correct.sum().item()
        if self.reduce == "mean":
            return correct.mean().item()
        if self.reduce is None:
            return correct.tolist()


class FusedAccuracyAtK:
    """Fuses AccuracyAtK to only perform .topk once"""

    def __init__(self, top_k: list, reduce="sum"):
        self.top_k = top_k
        self.reduce = reduce
        self.max_k = max(top_k)

    def __call__(self, logits, labels):
        """Yields topk accuracy for each self.top_k"""
        pred = logits.topk(self.max_k, -1, True, True).indices
        for k in self.top_k:
            correct = (pred[:, :k] == labels.unsqueeze(-1)).any(1).float()
            if self.reduce == "sum":
                yield k, correct.sum().item()
            if self.reduce == "mean":
                yield k, correct.mean().item()
            if self.reduce is None:
                yield k, correct.tolist()


class PredMetrics:
    def __init__(
        self, pred_times: Optional[List], prediction_windows: list, decimals=4
    ):
        self.pred_times = pred_times
        self.prediction_windows = prediction_windows
        self.decimals = decimals

    def __call__(self, logits, labels):
        assert logits.shape == labels.shape
        output = {}
        mask = labels != -100
        if self.pred_times is None and len(self.prediction_windows) == 1:
            mask = labels != -100
            output = self.compute(logits[mask], labels[mask])
        else:
            if self.pred_times is not None:
                for i, prediction_time in enumerate(self.pred_times):
                    preds, targets = logits[:, i], labels[:, i]
                    mask = targets != -100
                    metrics = self.compute(preds[mask], targets[mask])
                    for key, v in metrics.items():
                        output[f"{key}_time{prediction_time}"] = v

            if len(self.prediction_windows) > 1:
                for i, pred_window in enumerate(self.prediction_windows):
                    preds, targets = logits[:, :, i], labels[:, :, i]
                    mask = targets != -100
                    metrics = self.compute(preds[mask], targets[mask])
                    for key, v in metrics.items():
                        output[f"{key}_window{pred_window}"] = v

            keys = set([key.split("_")[0] for key in output])
            for key in keys:
                vals = [v for k, v in output.items() if k.startswith(key)]
                output[f"{key}_mean"] = sum(vals) / len(vals)

        for key, val in output.items():
            output[key] = float(round(val, self.decimals))
        return output

    def compute(self, logits, labels):
        raise NotImplementedError


class CustomROCS(PredMetrics):
    def compute(self, logits, labels):
        metrics = {}
        metrics["AUROC"] = roc_auc_score(labels, logits)
        metrics["PRAUC"] = average_precision_score(labels, logits)
        return metrics


class BestAtThreshold(PredMetrics):
    def __init__(self, *args, thresholds=torch.arange(0.1, 1, 0.1), **kwargs):
        super().__init__(*args, **kwargs)
        self.thresholds = thresholds

    def compute(self, logits, labels):
        threshold = self.best_threshold(logits, labels)
        tn, fp, fn, tp = confusion_matrix(labels, logits > threshold).ravel().tolist()
        metrics = {}
        metrics["threshold"] = threshold
        metrics["precision"] = tp / (tp + fp + 1e-9)
        metrics["recall"] = tp / (tp + fn + 1e-9)
        metrics["F1-score"] = (2 * metrics["precision"] * metrics["recall"]) / (
            metrics["precision"] + metrics["recall"] + 1e-9
        )

        metrics["TN"] = tn
        metrics["FP"] = fp
        metrics["FN"] = fn
        metrics["TP"] = tp
        return metrics

    def best_threshold(self, logits, labels):
        max_ = (None, -1)
        for threshold in self.thresholds:
            f1 = f1_score(labels, logits > threshold)
            if f1 > max_[1]:
                max_ = (threshold.item(), f1)
        return max_[0]
