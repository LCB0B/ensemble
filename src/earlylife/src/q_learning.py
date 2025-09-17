import copy
from collections import defaultdict

import lightning.pytorch as pl
import torch
import torch.nn as nn

from src.earlylife.src.encoder_nano_risk import RiskNanoEncoder

NUM_ACTIONS = 2


class CircularReplayBuffer:
    """ReplayBuffer for experiences using a circular ring buffer with pre-allocated tensors"""

    def __init__(self, capacity, input_dim, device):
        self.capacity = capacity
        self.device = device

        # Pre-allocate tensors
        self.states = torch.zeros(
            (capacity, input_dim), dtype=torch.float32, device=device
        )
        self.actions = torch.zeros(capacity, dtype=torch.int64, device=device)
        self.reward = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros(
            (capacity, input_dim), dtype=torch.float32, device=device
        )
        self.dones = torch.zeros(capacity, dtype=torch.int8, device=device)

        # Position trackers
        self.head = 0
        self.size = 0

    def add(self, states, actions, reward, next_states, dones):
        """Saves the experiences as individual transitions"""
        batch_size = states.size(0)

        if batch_size > self.capacity:
            raise ValueError(
                f"batch_size {batch_size} > capacity {self.capacity}, please raise capacity"
            )

        space_until_end = self.capacity - self.head
        if batch_size <= space_until_end:
            self.states[self.head : self.head + batch_size] = states
            self.actions[self.head : self.head + batch_size] = actions
            self.reward[self.head : self.head + batch_size] = reward
            self.next_states[self.head : self.head + batch_size] = next_states
            self.dones[self.head : self.head + batch_size] = dones

            self.head = (self.head + batch_size) % self.capacity
        else:  # Need to handle wrap-around
            # First part - fill until end
            self.states[self.head :] = states[:space_until_end]
            self.actions[self.head :] = actions[:space_until_end]
            self.reward[self.head :] = reward[:space_until_end]
            self.next_states[self.head :] = next_states[:space_until_end]
            self.dones[self.head :] = dones[:space_until_end]

            # Second part - wrap around to start
            remain = batch_size - space_until_end
            self.states[:remain] = states[space_until_end:]
            self.actions[:remain] = actions[space_until_end:]
            self.reward[:remain] = reward[space_until_end:]
            self.next_states[:remain] = next_states[space_until_end:]
            self.dones[:remain] = dones[space_until_end:]

            self.head = remain
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        """Returns batch_size samples from saved buffer"""
        if batch_size > self.size:
            raise ValueError("batch_size > self.size, cant sample batch_size")
        indices = torch.randint(0, self.size, (batch_size,))
        # indices = torch.multinomial(
        #     torch.softmax(self.states[: self.size].squeeze(-1), dim=0), batch_size
        # )
        # print(
        #     torch.quantile(
        #         self.states[indices].squeeze(-1),
        #         torch.tensor(
        #             [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1], device=self.states.device
        #         ),
        #     )
        # )
        # indices = torch.randint(0, self.size, (batch_size,))

        return (
            self.states[indices],
            self.actions[indices],
            self.reward[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.size


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
        grid_tokens = [(row == 1).nonzero()[:, 0] for row in batch["og_event"]]
        grid_abspos = torch.nn.utils.rnn.pad_sequence(
            [row[grid] for row, grid in zip(batch["og_abspos"], grid_tokens)],
            batch_first=True,
            padding_value=torch.inf,
        )
        censor = batch["censor"]
        # valid_targets = batch["target"][:, : grid_abspos.size(1), 1] # CHANGED!
        valid_targets = batch["target"][:, : grid_abspos.size(1), 1]

        return grid_abspos, censor, valid_targets

    def on_validation_epoch_end(self):
        log_dict = {}
        edt = torch.tensor(self.val_metrics["edt"])
        # print(edt[~edt.isnan()].sum(), sum(self.val_metrics["screens"]))
        edt = edt[~edt.isnan()].mean()
        num_screens = sum(self.val_metrics["screens"]) / sum(
            self.val_metrics["n_years"]
        )

        log_dict["edt"] = edt
        log_dict["num_screens"] = num_screens
        self.log_dict(log_dict, on_epoch=True)
        self.logger.experiment.add_scalar("result", edt, num_screens * 1000)
        print(edt, num_screens)

        self.val_metrics = defaultdict(list)

    def get_eval_metrics(self, actions, batch):
        """Returns the edt and num_screens"""
        grid_abspos, censor, valid_targets = self.get_prediction_inputs(batch)

        # Get EDT
        diff = (censor.unsqueeze(-1) - grid_abspos) / self.one_month_abspos
        diff[diff > self.pred_window] = -torch.inf  # Filter too early
        diff[(actions == 0)] = -torch.inf  # No EDT for no actions

        # Adjust EDT and indices for negative people and missed cancers
        edt, indices = diff.max(dim=-1)
        indices[edt == -torch.inf] = valid_targets[edt == -torch.inf].argmin(dim=-1) - 1
        indices[censor.isnan()] = valid_targets[censor.isnan()].argmin(dim=-1) - 1
        indices[indices == -1] = actions.size(1)  # If of no padding (^ .argmin fails)
        edt[edt == -torch.inf] = -18
        edt[censor.isnan()] = torch.nan

        # Get terminal states
        row_indices = torch.arange(actions.size(1), device=self.device).unsqueeze(0)
        terminal = row_indices > indices.unsqueeze(-1)

        return edt, terminal


class AgeScreeningModel(BaseScreeningModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.screening_age_lower = self.hparams.screening_age_lower
        self.screening_age_upper = self.hparams.screening_age_upper

    def get_val_action(self, states):
        """Returns the actions"""
        actions = torch.zeros_like(states)

        actions[:, self.screening_age_lower : self.screening_age_upper] = 1

        return actions

    def get_states(self, batch):
        argmin = batch["target"].argmin(dim=1).max()
        return batch["target"][:, :argmin, 1]


class RiskScreeningModel(BaseScreeningModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.risk_threshold = self.hparams.risk_threshold

        self.risk_model = RiskNanoEncoder(**self.hparams).eval()

    def get_val_action(self, states):
        """Returns the actions"""
        actions = torch.zeros_like(states)
        at_risk = states >= self.risk_threshold
        actions[at_risk] = 1

        return actions

    def get_states(self, batch):
        with torch.no_grad():
            x = self.risk_model.risk_step(batch, None)
            x = x[:, :, 1]
        # argmin = (
        #     batch["target"][:, :, 1].argmin(dim=1).max()
        # )  # only take relevant states
        # x = x[:, :argmin]  # Filter out most padding # CHANGED!
        x = torch.sigmoid(x)
        return x


class QScreeningModel(RiskScreeningModel):
    """Base class"""

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(self.hparams.d_screen, self.hparams.d_model),
            nn.Linear(self.hparams.d_model, NUM_ACTIONS),
        )
        self.target = copy.deepcopy(self.model)
        self.criterion = nn.MSELoss()

        self.epsilon = self.hparams["epsilon"]
        self.epsilon_end = self.hparams["epsilon_end"]
        self.epsilon_decay = self.hparams["epsilon_decay"]
        self.gamma = self.hparams["gamma"]

        self.alpha = self.hparams.get("alpha")

        self.buffer = CircularReplayBuffer(
            self.hparams.replay_size, self.hparams.d_screen, torch.device("cuda")
        )

    def configure_optimizers(self):
        for param in self.risk_model.parameters():
            param.requires_grad = False
        for param in self.target.parameters():
            param.requires_grad = False

        model_params = [
            param for param in self.model.parameters() if param.requires_grad
        ]
        model_optimizer = torch.optim.AdamW(
            model_params,
            lr=torch.tensor(self.hparams.learning_rate),
            betas=(self.hparams.beta1, self.hparams.beta2),
            fused=True,
        )

        return model_optimizer

    def get_states(self, batch):
        states = super().get_states(batch)
        return states.unsqueeze(-1)

    def get_action(self, states):
        """Returns the epsilon-greedy actions"""
        # with torch.no_grad(): # TODO: Cant make this work??
        action_values = self.model(states).argmax(dim=-1).detach()
        random_actions = torch.randint(
            0, NUM_ACTIONS, size=action_values.shape, device=self.device
        )
        exploration_mask = (
            torch.rand(action_values.shape, device=self.device) < self.epsilon
        )
        actions = torch.where(exploration_mask, random_actions, action_values)
        return actions

    def get_val_action(self, states):
        with torch.no_grad():
            q_values = self.model(states)
            actions = torch.argmax(q_values, dim=-1)
        return actions

    def forward(self, batch, batch_idx):
        """Forward pass with sampling AND update phase"""
        # DQN update
        if (batch_idx + 1) % 300 == 0:
            self.update_target_network()

        self.replay_forward(batch)

        loss = self.update_forward(batch["attn_mask"].size(0) * 20)

        return loss

    def replay_forward(self, batch):
        # Sampling phase
        states = self.get_states(batch)
        actions = self.get_action(states)
        reward, terminal = self.get_reward(states, actions, batch)
        next_states, dones = self.get_next_states(states, terminal)

        valid_episodes = ~terminal
        states = states.transpose(0, 1)[valid_episodes.T]
        actions = actions.T[valid_episodes.T]
        reward = reward.T[valid_episodes.T]
        next_states = next_states.transpose(0, 1)[valid_episodes.T]
        dones = dones.T[valid_episodes.T]

        self.buffer.add(states, actions, reward, next_states, dones)

    def update_forward(self, sample_size):
        (
            sampled_states,  # (N_T, 1)
            sampled_actions,  # (N_T)
            sampled_reward,  # (N_T)
            sampled_next_states,  # (N_T, 1)
            sampled_dones,  # (N_T)
        ) = self.buffer.sample(sample_size)

        rewards = sampled_reward - (self.alpha * sampled_actions)

        current_q = (
            self.model(sampled_states)
            .gather(-1, sampled_actions.unsqueeze(-1))
            .squeeze(-1)
        )  # Squeeze reward dim

        with torch.no_grad():
            next_q = self.target(sampled_next_states).max(-1)[0]
            target_q = rewards.squeeze(-1) + self.gamma * next_q * sampled_dones
        loss = self.criterion(current_q, target_q)
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss = self.forward(batch, batch_idx)
        self.log("train/loss", loss)
        self.log("epsilon", self.epsilon)

        return loss

    def get_reward(self, states, actions, batch):
        grid_abspos, censor, valid_targets = self.get_prediction_inputs(batch)
        reward = torch.zeros_like(actions, dtype=torch.float32)

        negative = True
        positive = True
        for i, c in enumerate(censor):
            if negative and torch.isnan(c):
                count = (grid_abspos[i] != torch.inf).sum()
                print("neg", states[i][:count].squeeze(-1))
                negative = False
            if positive and ~torch.isnan(c):
                count = (grid_abspos[i] != torch.inf).sum()
                print("pos", states[i][:count].squeeze(-1))
                positive = False
        print()

        # Get EDT
        diff = (censor.unsqueeze(-1) - grid_abspos) / self.one_month_abspos
        diff[diff > self.pred_window] = -torch.inf  # Filter too early
        diff[(actions == 0)] = -torch.inf  # No EDT for no actions

        # Adjust EDT and indices for negative people and missed cancers
        edt, indices = diff.max(dim=-1)
        indices[edt == -torch.inf] = valid_targets[edt == -torch.inf].argmin(dim=-1) - 1
        indices[censor.isnan()] = valid_targets[censor.isnan()].argmin(dim=-1) - 1
        edt[edt == -torch.inf] = -18
        edt[censor.isnan()] = 0

        # Get terminal states
        row_indices = torch.arange(actions.size(1), device=self.device).unsqueeze(0)
        terminal = row_indices > indices.unsqueeze(-1)

        reward[torch.arange(len(indices)), indices] = edt
        reward /= 18

        # Propagate reward to 2nd last state
        # second_last_screened = actions[torch.arange(len(indices)), indices - 1] == 1
        # reward[torch.arange(len(indices)), indices - 1][second_last_screened] = edt[
        #     second_last_screened
        # ] / (18 * 2)

        # Reward screening based on risk (only positives)
        # positives = ~censor.isnan()
        # screened = actions == 1
        # reward[positives.unsqueeze(-1) & screened] += (
        #     states[positives.unsqueeze(-1) & screened].squeeze(-1) / 10
        # )

        return reward, terminal

    def get_next_states(self, states, terminal):
        """Returns (next_states, dones)"""
        # Next states
        next_states = states.roll(-1)
        # Dones (to cancel out final state)
        dones = torch.ones_like(terminal)
        dones[torch.arange(dones.size(0)), terminal.float().argmax(dim=-1) - 1] = 0
        return next_states, dones

    def update_target_network(self):
        self.target = copy.deepcopy(self.model)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


class EnvelopeQScreeningModel(QScreeningModel):
    """Envelope Q-learning for screening policies"""

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(self.hparams.d_screen + 1, self.hparams.d_model),
            nn.Linear(self.hparams.d_model, NUM_ACTIONS),
        )
        self.target = copy.deepcopy(self.model)

        self.eval_pref = [1, 0.1, 0.01, 0.001]
        self.homotopy_increase_rate = self.hparams["homotopy_increase_rate"]
        self.homotopy = 0.0

    def get_val_action(self, states, preferences):
        with torch.no_grad():
            q_values = preferences * self.q(self.model, states, preferences)
            actions = torch.argmax(q_values, dim=-1)
        return actions

    def get_action(self, states):
        """Returns the epsilon-greedy actions WITH preferences"""
        preferences = (
            self.sample_preferences(states.shape[:1])
            .unsqueeze(-1)  # Adds timestep dimension
            .expand(-1, states.size(1))  # same preference for a person
            .unsqueeze(-1)  # Adds hidden_dim
        )

        action_values = (
            (preferences * self.q(self.model, states, preferences))
            .argmax(dim=-1)
            .detach()
        )
        random_actions = torch.randint(
            0, NUM_ACTIONS, size=action_values.shape, device=self.device
        )
        exploration_mask = (
            torch.rand(action_values.shape, device=self.device) < self.epsilon
        )
        actions = torch.where(exploration_mask, random_actions, action_values)
        return actions

    def sample_preferences(self, size):
        """Samples preference vector from hard-coded distribution"""
        dist = torch.distributions.uniform.Uniform(0, 0.1)
        return dist.sample(size).to(self.device)

    def q(self, model, states, preferences):
        concat_states = torch.cat([states, preferences], dim=-1)
        q_vals = model(concat_states)
        return q_vals

    def update_forward(self, sample_size):
        (
            sampled_states,  # (N_T, 1)
            sampled_actions,  # (N_T)
            sampled_reward,  # (N_T)
            sampled_next_states,  # (N_T, 1)
            sampled_dones,  # (N_T)
        ) = self.buffer.sample(sample_size)
        N_T = sample_size
        N_W = 10

        sampled_preferences = self.sample_preferences((N_W, 1))  # (N_W)

        exp_x = sampled_states.unsqueeze(0).expand(N_W, N_T, -1)
        exp_nx = sampled_next_states.unsqueeze(0).expand(N_W, N_T, -1)
        exp_pref = sampled_preferences.unsqueeze(1).expand(N_W, N_T, -1)
        exp_a = sampled_actions.expand(N_W, N_T).unsqueeze(-1)  # (N_W, N_T, 1)
        exp_reward = sampled_reward.unsqueeze(0).unsqueeze(-1)  # (1, N_T, 1)

        exp_rewards = exp_reward - (exp_pref * exp_a)  # (N_W, N_T, 1)

        current_q = (
            self.q(self.model, exp_x, exp_pref).gather(-1, exp_a).squeeze(-1)
        )  # Squeeze reward dim

        with torch.no_grad():
            next_q, pref_idxs = (
                (exp_pref * self.q(self.target, exp_nx, exp_pref)).max(-1)[0].max(dim=0)
            )
            max_pref_rewards = exp_rewards[pref_idxs, torch.arange(exp_rewards.size(1))]
            target_q = (
                (max_pref_rewards.squeeze(-1) + self.gamma * next_q * sampled_dones)
                .unsqueeze(0)
                .expand(N_W, N_T)
            )
        lossA = self.criterion(current_q, target_q)
        lossB = self.criterion(
            (current_q * exp_pref.squeeze(-1)), (target_q * exp_pref.squeeze(-1))
        )
        loss = (1 - self.homotopy) * lossA + self.homotopy * lossB
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss = super().training_step(batch, batch_idx)
        self.log("homotopy", self.homotopy)

        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:  # TODO: REDO
        states = self.get_states(batch).unsqueeze(-1)

        for w in self.eval_pref:
            preferences = torch.zeros(states.shape, device=self.device) + w

            actions = self.get_val_action(states, preferences)

            edt, terminal = self.get_eval_metrics(actions, batch)
            actions[terminal] = 0

            self.val_metrics[f"edt_{w}"].extend(edt.tolist())
            self.val_metrics[f"screens_{w}"].extend(actions.sum(dim=-1).tolist())
            self.val_metrics["n_years"].extend((~terminal).sum(dim=-1).tolist())

    def on_validation_epoch_end(self):
        log_dict = {}
        for w in self.eval_pref:
            edt = torch.tensor(self.val_metrics[f"edt_{w}"])
            edt = edt[~edt.isnan()].mean()
            num_screens = sum(self.val_metrics[f"screens_{w}"]) / sum(
                self.val_metrics["n_years"]
            )

            log_dict[f"edt_{w}"] = edt
            log_dict[f"num_screens_{w}"] = num_screens
        print(log_dict)
        self.log_dict(log_dict, on_epoch=True)
        self.logger.experiment.add_scalar("result", edt, num_screens * 1000)

        self.val_metrics = defaultdict(list)

    def update_target_network(self):
        super().update_target_network()
        self.homotopy = min(1.0, self.homotopy + self.homotopy_increase_rate)
        self.homotopy *= 0.95
