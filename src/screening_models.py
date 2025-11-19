import torch
import lightning.pytorch as pl
from collections import defaultdict
from src.encoder_nano_risk import (
    RiskPredictionFinetuneNanoEncoder,
    FamilyRiskPredictionFinetuneNanoEncoder,
)


class BaseScreeningModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.val_metrics = defaultdict(list)

        self.one_month_abspos = (365.25 * 24) / 12
        self.pred_window = 18

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        states = self.get_states(batch)
        actions = self.get_val_action(states)

        edt, terminal = self.get_eval_metrics(actions, batch)
        actions[terminal] = 0

        self.val_metrics["edt"].extend(edt.tolist())
        self.val_metrics["screens"].extend(actions.sum(dim=-1).tolist())
        self.val_metrics["n_years"].extend((~terminal).sum(dim=-1).tolist())

    def get_prediction_inputs(self, batch):
        grid_abspos = torch.nn.utils.rnn.pad_sequence(
            [
                row[grid]
                for row, grid in zip(batch["og_abspos"], batch["predict_tokens"])
            ],
            batch_first=True,
            padding_value=torch.inf,
        )
        outcome = batch["outcome"]
        valid_targets = batch["target"][:, :, 1]

        return grid_abspos, outcome, valid_targets

    def on_validation_epoch_end(self):
        log_dict = {}
        edt = torch.tensor(self.val_metrics["edt"])
        edt = edt[~edt.isnan()].mean()
        num_screens = sum(self.val_metrics["screens"]) / sum(
            self.val_metrics["n_years"]
        )

        log_dict["edt"] = edt
        log_dict["num_screens"] = num_screens
        self.log_dict(log_dict, on_epoch=True)
        self.logger.experiment.add_scalar("result", edt, num_screens * 1000)

        self.val_metrics = defaultdict(list)

    def get_eval_metrics(self, actions, batch):
        """Returns the edt and num_screens"""
        grid_abspos, outcome, valid_targets = self.get_prediction_inputs(batch)

        # Get EDT
        diff = (outcome.unsqueeze(-1) - grid_abspos) / self.one_month_abspos
        diff[diff > self.pred_window] = -torch.inf  # Filter too early
        diff[(actions == 0)] = -torch.inf  # No EDT for no actions

        # Adjust EDT and indices for negative people and missed cancers
        edt, indices = diff.max(dim=-1)
        indices[edt == -torch.inf] = valid_targets[edt == -torch.inf].argmin(dim=-1) - 1
        indices[outcome.isnan()] = valid_targets[outcome.isnan()].argmin(dim=-1) - 1
        indices[indices == -1] = actions.size(1)  # If no padding (^ .argmin fails)
        edt[edt == -torch.inf] = -18
        edt[outcome.isnan()] = torch.nan

        # Get terminal states
        row_indices = torch.arange(actions.size(1), device=indices.device).unsqueeze(0)
        terminal = row_indices > indices.unsqueeze(-1)

        return edt, terminal


class RandomScreeningModel(BaseScreeningModel):
    def get_val_action(self, states):
        actions = torch.zeros_like(states)
        actions[torch.rand_like(actions) > 0.5] = 1
        return actions

    def get_states(self, batch):
        return batch["target"][:, :, 1]


class AgeScreeningModel(BaseScreeningModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.screening_age_lower = self.hparams.screening_age_lower
        self.screening_age_upper = self.hparams.screening_age_upper
        if self.screening_age_upper is None:
            self.screening_age_upper = 1000

    def get_val_action(self, states):
        """Returns the actions"""
        actions = torch.zeros_like(states)
        at_age = (self.screening_age_lower <= states) & (
            states <= self.screening_age_upper
        )
        actions[at_age] = 1
        return actions

    def get_states(self, batch):
        grid_age = torch.nn.utils.rnn.pad_sequence(
            [row[mask] for row, mask in zip(batch["og_age"], batch["predict_tokens"])],
            batch_first=True,
            padding_value=torch.inf,
        )
        return grid_age


class RiskScreeningModel(BaseScreeningModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.risk_threshold = self.hparams.risk_threshold
        self.time_window = 1

        self.risk_model = RiskPredictionFinetuneNanoEncoder(**self.hparams).eval()

    def get_val_action(self, states):
        """Returns the actions"""
        actions = torch.zeros_like(states)
        at_risk = states >= self.risk_threshold
        actions[at_risk] = 1

        return actions

    def get_states(self, batch):
        with torch.no_grad():
            x = self.risk_model.predict_step(batch, None)
            x = x[:, :, self.time_window]

        return x

    def predict_step(self, batch, batch_idx):
        x = self.risk_model.predict_step(batch, None)
        x = x[:, :, self.time_window]
        return x


class FamilyRiskScreeningModel(RiskScreeningModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.risk_model = FamilyRiskPredictionFinetuneNanoEncoder(**self.hparams).eval()
