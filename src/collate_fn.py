""" Implements all collate functionality """

from typing import Dict, List, Tuple, Union, Literal

import torch

from src.collate_utils import (
    calc_age_from_abspos,
    censor_data,
    censor_person,
    combine_and_sort_family_seq,
    mask_inputs,
)


class Collate:
    """Implements the core collate functions"""

    def __init__(self, max_seq_len=512, background_length=0, **kwargs):
        self.max_seq_len = max_seq_len
        self.truncate_length = max_seq_len - background_length
        self.background_length = background_length
        self.bg_events = int(background_length > 0)
        self.data_keys = ["event", "age", "abspos", "segment"]

    def __call__(self, batch: Dict[str, List]) -> dict:
        return self.main(batch)

    def main(self, batch: Dict[str, List[dict]]) -> dict:
        """The main processing function"""
        output = {key: [] for key in self.data_keys}

        for person in batch["data"]:
            indv = self.process_person(person)
            for key, v in indv.items():
                output[key].append(v)

        # Pad all person keys
        for key in self.data_keys:
            output[key] = self._pad(output[key])

        return output

    def process_person(self, person, truncate_length: int = None):
        """Handles all person-specific processsing"""
        output = {}
        truncate_length = (
            truncate_length if truncate_length is not None else self.truncate_length
        )
        # Start with events
        person_seq, person_event_lens, event_border = self._flatten(
            person.pop("event"),
            truncate_length=truncate_length,
        )
        output["event"] = person_seq

        person["segment"] = list(range(1, len(person_event_lens) + 1))
        # Add rest of keys
        repeats = torch.as_tensor(person_event_lens)
        for key in person:
            truncated_seq = self._truncate(person[key], event_border)
            expanded_seq = self.expand(truncated_seq, repeats)
            output[key] = expanded_seq
        return output

    def _truncate(self, seq: list, event_border: int):
        if event_border is None:
            return seq
        return seq[: self.bg_events] + seq[event_border:]

    @staticmethod
    def expand(seq: list, repeats: torch.Tensor) -> torch.Tensor:
        """Repeats seq[i] repeats[i] times"""
        dtype = torch.int32 if isinstance(seq[0], int) else torch.float32
        return torch.repeat_interleave(torch.as_tensor(seq, dtype=dtype), repeats)

    def _flatten(self, events: List[List[int]], truncate_length: int):
        """Flattens events and (optional) truncates, returning flatten_seq and the last event idx"""
        person_seq, person_event_lens, event_border = (
            self._flatten_reverse_and_truncate(events, truncate_length)
        )
        return person_seq, person_event_lens, event_border

    def _flatten_reverse_and_truncate(
        self, sequence: List, truncate_length: int
    ) -> Tuple[list, list, int]:
        """Flattens a reversed list (keeping newest info) until truncate_length reached, adds background and then returns event_idx (if terminated) and list"""
        result, event_lens = [], []
        total_length = 0
        for i, sublist in enumerate(reversed(sequence)):
            n = len(sublist)
            total_length += n
            if total_length > truncate_length:
                break
            event_lens.append(n)
            result.extend(sublist[::-1])
        else:  # If loop finished (total_length < truncate_length)
            return result[::-1], event_lens[::-1], None

        # Add background onto it
        for sublist in reversed(sequence[: self.bg_events]):
            result.extend(sublist[::-1])
            event_lens.append(len(sublist))
        return result[::-1], event_lens[::-1], -i

    def _pad(
        self,
        sequence: Union[list, torch.Tensor],
        dtype: torch.dtype = None,
        **pad_kwargs,
    ) -> torch.Tensor:
        """Pads the sequence (using pad_kwargs) converts to tensor"""
        # Convert to tensors and pad with dtype conversion
        if not torch.is_tensor(sequence[0]) or dtype != sequence[0].dtype:
            sequence = [torch.as_tensor(seq, dtype=dtype) for seq in sequence]
        return torch.nn.utils.rnn.pad_sequence(
            sequence, batch_first=True, **pad_kwargs
        ).to(dtype)


class MaskCollate(Collate):
    """Standard collate with masking"""

    def __init__(
        self,
        *args,
        vocab: dict,
        mask_prob=0.15,
        replace_prob=0.8,
        random_prob=0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vocab = vocab
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_prob = random_prob

    def __call__(self, batch: Dict[str, List]) -> dict:
        output = self.main(batch)

        # Mask events and produce targets
        output["event"], output["target"] = mask_inputs(
            output["event"],
            self.vocab,
            mask_prob=self.mask_prob,
            replace_prob=self.replace_prob,
            random_prob=self.random_prob,
        )
        return output


class AutoregressiveCollate(Collate):
    """Standard collate with Autoregressive shift"""

    def __call__(self, batch):
        output = self.main(batch)

        output["target"] = output["event"][:, 1:]
        output["target"] = output["target"].masked_fill(output["target"] == 0, -100)
        for feature in ["event", "abspos", "age", "segment"]:
            output[feature] = output[feature][:, :-1]
        return output


class CensorCollate(Collate):
    """Standard collate with censoring"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, batch: Dict[str, List]) -> dict:
        """Input: List of Batch, Outcomes"""
        batch["data"] = self._censor_data(batch)

        output = self.main(batch)
        output = self._add_outcome_info(output, batch["outcome_info"])
        return output

    def _add_outcome_info(self, output: dict, outcome_info: List[dict]) -> Dict:
        output.update(
            {
                "target": [out["target"] for out in outcome_info],
                "person_id": [out["person_id"] for out in outcome_info],
            }
        )
        output["target"] = self._adjust_targets(output["target"])
        return output

    def _adjust_targets(self, targets):
        return torch.as_tensor(targets, dtype=torch.float32)

    def _censor_data(self, batch):
        return censor_data(
            batch["data"], batch["outcome_info"], background=self.bg_events
        )


class PredictCollate(Collate):
    """Standard collate with PREDICT tokens"""

    def __init__(self, *args, predict_token_id: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_token_id = predict_token_id

    def main(self, batch):
        for person, outcome in zip(batch["data"], batch["outcome_info"]):
            person["predict"] = outcome["predict"]
        return super().main(batch)

    def process_person(self, person, truncate_length: int = None):
        """Handles all person-specific processsing"""
        output = {}
        truncate_length = (
            truncate_length if truncate_length is not None else self.truncate_length
        )

        # Start with events
        person_seq, person_event_lens, event_border = self._flatten(
            person["event"],
            truncate_length=truncate_length - len(person["predict"]),
        )

        # Add rest of keys
        for key in self.data_keys:
            if key == "segment":
                continue
            truncated_seq = self._truncate(person[key], event_border)
            output[key] = truncated_seq

        output["event_lens"] = person_event_lens
        output = self.add_predict_tokens_to_person(output, person["predict"])
        person_event_lens = output.pop("event_lens")

        output["segment"] = list(range(1, len(person_event_lens) + 1))

        # Expand in the end (easier grid token sort and segment creation)
        repeats = torch.as_tensor(person_event_lens)
        for key in output:
            if key == "event":
                output[key] = torch.as_tensor(sum(output[key], []), dtype=torch.int32)
            else:
                output[key] = self.expand(output[key], repeats)
        return output

    def add_predict_tokens_to_person(self, person, predictions):
        """Adds a PREDICT token (self.predict_token_id used) to the person for every prediction"""
        person = self.add_predict_features_to_person(person, predictions)

        sorted_idxs = sorted(
            range(len(person["abspos"])), key=lambda i: person["abspos"][i]
        )

        return {key: [value[i] for i in sorted_idxs] for key, value in person.items()}

    def add_predict_features_to_person(self, person, predictions):
        """Adds predict-related features to person"""
        person["event"].extend([[self.predict_token_id] for _ in predictions])
        person["abspos"].extend(predictions)
        person["age"].extend(
            [calc_age_from_abspos(person["abspos"][0], a) for a in predictions]
        )
        person["event_lens"].extend([1] * len(predictions))
        return person


class AutoregressivePredictCollate(PredictCollate, AutoregressiveCollate):
    """Autoregressive with Predict tokens"""

    pass


class PredictCensorCollate(PredictCollate, CensorCollate):
    """Standard collate with censoring"""

    def __init__(
        self,
        *args,
        prediction_windows: list,
        padding_side: Literal["left", "right"] = "left",
        **kwargs,
    ):
        PredictCollate.__init__(self, *args, **kwargs)
        CensorCollate.__init__(self, *args, **kwargs)
        self.prediction_windows = torch.as_tensor(prediction_windows)
        self.padding_side = padding_side

    def _adjust_targets(self, targets):
        return self._pad(
            targets,
            dtype=torch.float32,
            padding_value=-100,
            padding_side=self.padding_side,
        )

    def _add_outcome_info(self, output: dict, outcome_info: List[dict]) -> Dict:
        outcome_info = self._create_targets(outcome_info)
        output["outcome"] = torch.tensor([out["outcome"] for out in outcome_info])
        return super()._add_outcome_info(output, outcome_info)

    def _create_targets(
        self,
        outcome_info: Tuple[Dict],
    ) -> List[Dict]:
        """Creates targets (k=prediction window) per predict abspos"""
        for outcome in outcome_info:
            predict_abspos = torch.as_tensor(sorted(outcome["predict"].tolist()))
            target = (
                predict_abspos.unsqueeze(-1) + self.prediction_windows
            ) >= outcome["outcome"]
            outcome["target"] = target
        return outcome_info


class FamilyCollate(Collate):
    """
    Family-aware collate that uses prediction windows for censoring.
    feature_set (list): List of family types to include (e.g. ["Child", "Mother"])
    """

    family_type = {"Child": 1, "Father": 2, "Mother": 3}

    def __init__(self, *args, feature_set: List[str], **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_set = feature_set
        self.process_set = feature_set
        self.data_keys += ["family_type"]

    def process_person(self, family):
        family_seq = []
        valid_family = [key for key in family if key in self.feature_set]
        truncate_length = self.max_seq_len // len(valid_family) - self.background_length
        for typ, person in family.items():
            if typ not in self.process_set:
                continue
            person["family_type"] = [self.family_type[typ]] * len(person["event"])
            if typ == "Child":
                indv = self.process_child(person, truncate_length)
            else:
                indv = self.process_parent(person, truncate_length)

            family_seq.append(indv)

        combined = combine_and_sort_family_seq(family_seq)
        return combined

    def process_child(self, person, truncate_length):
        return super().process_person(person, truncate_length)

    def process_parent(self, person, truncate_length):
        return super().process_person(person, truncate_length)


class FamilyPredictCensorCollate(FamilyCollate, PredictCensorCollate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "Child" not in self.process_set:
            self.process_set.append("Child")

    def main(self, batch):
        for family, outcome in zip(batch["data"], batch["outcome_info"]):
            family["Child"]["predict"] = outcome["predict"]

        return super().main(batch)

    def add_predict_features_to_person(self, person, predictions):
        person = super().add_predict_features_to_person(person, predictions)
        person["family_type"].extend([person["family_type"][0]] * len(predictions))
        return person

    def _censor_data(self, batch):
        """Censors each person in data using the info in each outcome_info"""
        censored_data = []
        for family, outcome in zip(batch["data"], batch["outcome_info"], strict=True):
            for typ, person in family.items():
                family[typ] = censor_person(person, outcome, background=self.bg_events)
            censored_data.append(family)
        return censored_data

    def process_child(self, person, truncate_length):
        if "Child" not in self.feature_set:
            person = self.remove_all_but_slice_events(person, slice(None, 1))
        indv = PredictCensorCollate.process_person(self, person, truncate_length)
        if "Child" not in self.feature_set:
            # Remove first event that we kept earlier
            indv = self.remove_all_but_slice_events(indv, slice(1, None))
        return indv

    def process_parent(self, person, truncate_length):
        return Collate.process_person(self, person, truncate_length)

    def remove_all_but_slice_events(self, person, _slice):
        """Removes information related to events based on _slice"""
        for key, value in self.data_keys.items():
            person[key] = value[_slice]
        if _slice == slice(None, 1):
            person["event"] = [[None]]  # Handle special event case
        elif _slice == slice(1, None):  # Adjust segment
            person["segment"] = [seg - 1 for seg in person["segment"]]
        return person


class FamilyPredictCensorRegressionCollate(FamilyPredictCensorCollate):
    """
    Family-aware collate that uses prediction windows for censoring and creates a regression target.
    """

    def _add_outcome_info(self, output: dict, outcome_info: List[dict]) -> Dict:
        outcome_info = self._create_regression_targets(outcome_info)
        output = super()._add_outcome_info(output, outcome_info)
        output["target_regression"] = torch.stack(
            [out["target_regression"] for out in outcome_info]
        )
        return output

    def _create_regression_targets(
        self,
        outcome_info: Tuple[Dict],
    ) -> List[Dict]:
        """Creates targets (k=prediction window) per predict abspos"""
        for outcome in outcome_info:
            target_regression = torch.as_tensor(outcome["outcome_regression"])

            outcome["target_regression"] = torch.repeat_interleave(
                target_regression, len(outcome["predict"].tolist())
            )

            # # TODO: Should we just save this instead ? Could also be used to subset preds etc, right now we infer n_predict_tokens based on first batch
            # outcome["n_predict_tokens"] = len(outcome["predict"].tolist())
        return outcome_info


class FamilyAutoregressiveCollate(FamilyCollate, AutoregressiveCollate):
    """Autoregressive with Family"""

    pass
