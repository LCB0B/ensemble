from sklearn.metrics import average_precision_score, roc_auc_score


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


class CustomROCS:
    def __init__(
        self, pred_times: list, prediction_windows: list, reduce=None, decimals=4
    ):
        self.pred_times = pred_times if isinstance(pred_times, list) else [pred_times]
        self.prediction_windows = prediction_windows
        self.reduce = reduce
        self.decimals = decimals

    def __call__(self, logits, labels):
        assert logits.shape == labels.shape
        output = {}
        mask = labels != -100
        if len(self.pred_times) == 1 and len(self.prediction_windows) == 1:
            output["AUROC"] = roc_auc_score(labels[mask], logits[mask])
            output["PRAUC"] = average_precision_score(labels[mask], logits[mask])
        else:
            if len(self.pred_times) > 1:
                for i, prediction_time in enumerate(self.pred_times):
                    preds, targets = logits[:, i], labels[:, i]
                    mask = targets != -100
                    output[f"AUROC_time{prediction_time}"] = roc_auc_score(
                        targets[mask], preds[mask]
                    )
                    output[f"PRAUC_time{prediction_time}"] = average_precision_score(
                        targets[mask], preds[mask]
                    )

            if len(self.prediction_windows) > 1:
                for i, pred_window in enumerate(self.prediction_windows):
                    preds, targets = (logits[:, :, i], labels[:, :, i])
                    mask = targets != -100
                    output[f"AUROC_window{pred_window}y"] = roc_auc_score(
                        targets[mask], preds[mask]
                    )
                    output[f"PRAUC_window{pred_window}y"] = average_precision_score(
                        targets[mask], preds[mask]
                    )

            aurocs = [v for k, v in output.items() if "AUROC" in k]
            praucs = [v for k, v in output.items() if "PRAUC" in k]
            output["AUROC_mean"] = sum(aurocs) / len(aurocs)
            output["PRAUC_mean"] = sum(praucs) / len(praucs)

        for key, val in output.items():
            output[key] = round(val, self.decimals)
        return output
