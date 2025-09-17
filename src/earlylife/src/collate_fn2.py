""" Implements all collate functionality """

import itertools
import math
from typing import Dict, List, Tuple, Union

import torch

from src.earlylife.src.collate_utils2 import (
    calc_age_from_abspos,
    censor_data,
    censor_person,
    combine_and_sort_family_seq,
    get_n_grid_tokens,
    mask_inputs,
)

ONE_YEAR_ABSPOS = 365.25 * 24


class Collate:
    """Implements the core collate functions"""

    def __init__(self, truncate_length=512, background_length=0, **kwargs):
        self.truncate_length = truncate_length
        self.background_length = background_length
        self.max_seq_len = truncate_length + background_length
        self.bg_events = int(self.background_length > 0)

    def __call__(self, batch: Dict[str, List]) -> dict:
        return self.main(batch)

    def main(self, batch: Dict[str, List[dict]]) -> dict:
        """The main processing function"""
        data = batch["data"]
        data_keys = self.data_keys(data[0])
        output = {key: [] for key in data_keys}

        for person in data:
            indv = self.process_person(person)
            for key, v in indv.items():
                output.setdefault(key, []).append(v)  # Allows arbitrary keys in indv

        # Pad all person keys
        for key in data_keys:
            output[key] = self._pad(output[key])

        return output

    def data_keys(self, person):
        """Returns the relevant data keys for the person and optionally segment"""
        return [key for key in person] + ["segment"]

    def process_person(self, person):
        """Handles all person-specific processsing"""
        output = {}
        # Start with events
        person_seq, person_event_lens, event_border = self._flatten(person.pop("event"))
        output["event"] = person_seq

        person["segment"] = list(range(1, len(person_event_lens) + 1))
        # Add rest of keys
        for key in person:
            truncated_seq = self._truncate(person[key], event_border)
            expanded_seq = self.expand(truncated_seq, person_event_lens)
            output[key] = expanded_seq

        return output

    def _truncate(self, seq: list, event_border: int):
        if event_border is None:
            return seq
        return seq[: self.bg_events] + seq[event_border:]

    @staticmethod
    def expand(seq: list, repeats: list) -> torch.Tensor:
        """Repeats seq[i] repeats[i] times"""
        return torch.repeat_interleave(torch.as_tensor(seq), torch.as_tensor(repeats))

    def _flatten(self, events: List[List[int]], truncate_length: int = None):
        """Flattens events and (optional) truncates, returning flatten_seq and the last event idx"""
        person_seq, person_event_lens, event_border = (
            self._flatten_reverse_and_truncate(
                events,
                (
                    truncate_length
                    if truncate_length is not None
                    else self.truncate_length
                ),
            )
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
        padding_value=0,
    ) -> torch.Tensor:
        """Pads the sequence (using padding_value) converts to tensor"""
        # Conver to tensors and get max_len and max_idx
        sequences = [torch.as_tensor(seq, dtype=dtype) for seq in sequence]

        return torch.nn.utils.rnn.pad_sequence(
            sequences,
            batch_first=True,
            padding_value=padding_value,
        )


class AutoregressiveCollate(Collate):
    def __call__(self, batch):
        output = self.main(batch)
        output["target"] = output["event"][:, 1:]
        for feature in ["event", "abspos", "age", "segment"]:
            output[feature] = output[feature][:, :-1]
        return output


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


class CensorCollate(Collate):
    def __call__(self, batch: Dict[str, List]) -> dict:
        """Input: List of Batch, Outcomes"""
        batch["data"] = self._censor_data(batch["data"], batch["outcome_info"])

        output = self.main(batch)
        output = self._add_outcome_info(output, batch["outcome_info"])
        return output

    def _add_outcome_info(self, output: dict, outcome_info: List[dict]) -> Dict:
        output.update(
            {
                "target": [out["target"] for out in outcome_info],
                "person_id": [out["person_id"] for out in outcome_info],
                "censor": torch.tensor([out["censor"] for out in outcome_info]),
            }
        )
        output["target"] = self._adjust_targets(output["target"])
        return output

    def _adjust_targets(self, targets):
        return torch.as_tensor(targets, dtype=torch.float32)

    def _censor_data(self, data, outcome_info):
        return censor_data(
            data,
            outcome_info,
            background=self.bg_events,
        )


class PredictCensorCollate(CensorCollate):
    """Standard collate with censoring"""

    def __init__(
        self, *args, prediction_windows: list, predict_token_id: int = 1, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.predict_token_id = predict_token_id
        self.prediction_windows = torch.as_tensor(prediction_windows)

    def main(self, batch: Dict[str, List]):
        # Initialize keys
        data_keys = data_keys = self.data_keys(batch["data"][0])
        output = {key: [] for key in data_keys}

        # Standard __call__ with new process_person
        for person, outcome in zip(batch["data"], batch["outcome_info"]):
            indv = self.process_person(person, outcome)
            for key, v in indv.items():
                output[key].append(v)

        # Pad all person keys
        for key in data_keys:
            output[key] = self._pad(output[key])

        return output

    def process_person(self, person, outcome):
        """Handles all person-specific processsing"""
        if not isinstance(outcome["predict"], list):
            outcome["predict"] = [outcome["predict"]]
        output = {}

        n_prediction_tokens = len(outcome["predict"])
        # Start with events
        person_seq, person_event_lens, event_border = self._flatten(
            person["event"],
            truncate_length=self.truncate_length - n_prediction_tokens,
        )

        # Add rest of keys
        for key in person:
            truncated_seq = self._truncate(person[key], event_border)
            output[key] = truncated_seq

        output["event_lens"] = person_event_lens  # Add event_lens for grid_tokens
        output = self.add_predict_tokens_to_person(output, outcome["predict"])
        person_event_lens = output.pop("event_lens")  # Not needed in output

        output["segment"] = list(range(1, len(person_event_lens) + 1))

        # Expand in the end (easier grid token sort and segment creation)
        for key in output:
            if key == "event":
                output[key] = sum(output[key], [])
            else:
                output[key] = self.expand(output[key], person_event_lens)

        return output

    def add_predict_tokens_to_person(self, person, predictions):
        """Adds a PREDICT token (self.predict_token_id used) to the person every prediction"""
        first_abspos = person["abspos"][0]
        person["event"].extend([[self.predict_token_id] for _ in predictions])
        person["abspos"].extend(predictions)
        person["age"].extend(
            [calc_age_from_abspos(first_abspos, a) for a in predictions]
        )
        person["event_lens"].extend([1] * len(predictions))

        sorted_idxs = sorted(
            range(len(person["abspos"])), key=lambda i: person["abspos"][i]
        )

        return {key: [value[i] for i in sorted_idxs] for key, value in person.items()}

    def _adjust_targets(self, targets):
        return self._pad(
            targets,
            dtype=torch.float32,
            padding_value=-100,
        )

    def _add_outcome_info(self, output: dict, outcome_info: List[dict]) -> Dict:
        outcome_info = self._create_targets(outcome_info)
        return super()._add_outcome_info(output, outcome_info)

    def _create_targets(
        self,
        outcome_info: Tuple[Dict],
    ) -> List[Dict]:
        """Creates targets (k=prediction window) per predict abspos"""
        for outcome in outcome_info:
            predict_abspos = torch.as_tensor(outcome["predict"])
            target = (
                predict_abspos.unsqueeze(-1) + self.prediction_windows
            ) >= outcome["outcome"]
            outcome["target"] = target
        return outcome_info


class GridCollate(Collate):
    """Standard collate with grid tokens"""

    def __init__(
        self,
        *args,
        predictions_per_year: float = 1,
        grid_token_id: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.grid_interval = ONE_YEAR_ABSPOS * predictions_per_year
        self.grid_token_id = grid_token_id

    def __call__(self, batch: Dict[str, List]) -> dict:
        return self.main(batch)

    def main(self, batch: Dict[str, List]):
        # Initialize keys
        data_keys = data_keys = self.data_keys(batch["data"][0])
        output = {key: [] for key in data_keys + ["sequence_lens"]}

        # Standard __call__ with new process_person
        for person, outcome in zip(batch["data"], batch["outcome_info"]):
            indv = self.process_person(person, self.get_last_valid_date(outcome))
            for key, v in indv.items():
                output[key].append(v)

            # Add padding information
            output["sequence_lens"].append(len(indv["event"]))

        # Pad all person keys
        for key in data_keys:
            output[key] = self._pad(output[key])
        output["sequence_lens"] = torch.as_tensor(output["sequence_lens"])

        return output

    def process_person(self, person, last_valid_date: float):
        """Handles all person-specific processsing"""
        output = {}
        # Start with events
        n_grid_tokens = get_n_grid_tokens(
            last_valid_date, person["abspos"][0], self.grid_interval
        )
        person_seq, person_event_lens, event_border = self._flatten(
            person["event"], truncate_length=self.truncate_length - n_grid_tokens
        )

        # Add rest of keys
        for key in person:
            truncated_seq = self._truncate(person[key], event_border)
            output[key] = truncated_seq

        output["event_lens"] = person_event_lens  # Add event_lens for grid_tokens
        output = self.add_grid_tokens_to_person(output, last_valid_date)
        person_event_lens = output.pop("event_lens")  # Not needed in output

        if self.segment:
            output["segment"] = list(range(1, len(person_event_lens) + 1))

        # Expand in the end (easier grid token sort and segment creation)
        for key in output:
            if key == "event":
                output[key] = sum(output[key], [])
            else:
                output[key] = self.expand(output[key], person_event_lens)

        return output

    def add_grid_tokens_to_person(self, person, last_valid_date: float):
        """Adds a GRID token (self.grid_token_id used) to the person every self.grid_interval"""
        if person["abspos"][0] > last_valid_date:
            abspos_range = torch.tensor([person["abspos"][0]])
        else:
            abspos_range = torch.arange(
                person["abspos"][0], last_valid_date, self.grid_interval
            )
        person["event"].extend([[self.grid_token_id] for _ in abspos_range])
        person["abspos"].extend(abspos_range.tolist())
        person["age"].extend([y for y in range(len(abspos_range))])
        person["event_lens"].extend([1] * len(abspos_range))

        sorted_idxs = sorted(
            range(len(person["abspos"])), key=lambda i: person["abspos"][i]
        )

        return {key: [value[i] for i in sorted_idxs] for key, value in person.items()}

    def get_last_valid_date(self, outcome_info):
        """Gets the last_valid_date for the outcome_info"""
        return outcome_info["event_final_date"]


class MaskGridCollate(MaskCollate, GridCollate):
    """MLM for GridCollate"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        GridCollate.__init__(self, *args, **kwargs)
        MaskCollate.__init__(self, *args, **kwargs)


class GridCensorCollate(CensorCollate, GridCollate):
    """CensorCollate with grid tokens and multi-trajectory targets"""

    def __init__(
        self,
        *args,
        prediction_windows: list = None,
        **kwargs,
    ):
        GridCollate.__init__(self, *args, **kwargs)
        CensorCollate.__init__(self, *args, **kwargs)
        self.prediction_windows = torch.as_tensor(prediction_windows)

    def _add_outcome_info(self, output: dict, outcome_info: List[dict]) -> Dict:
        outcome_info = self._create_targets(output, outcome_info)
        return super()._add_outcome_info(output, outcome_info)

    def _create_targets(
        self,
        batch: Dict,
        outcome_info: Tuple[Dict],
    ) -> List[Dict]:
        """Creates targets (k=prediction window) per person event"""
        for i, outcome in enumerate(outcome_info):
            grid_tokens = batch["event"][i] == self.grid_token_id
            grid = batch["abspos"][i][grid_tokens]
            outcome["target"] = self.construct_targets_grid(grid, outcome)
        return outcome_info

    def construct_targets_grid(self, grid, outcome):
        """Creates targets (k=prediction window) per abspos"""
        grid = torch.as_tensor(grid)
        if outcome["outcome"]:
            return (grid.unsqueeze(1) + self.prediction_windows) >= outcome["censor"]
        else:
            return torch.zeros(len(grid), len(self.prediction_windows))

    def _adjust_targets(self, targets):
        return self._pad(
            targets,
            dtype=torch.float32,
            padding_value=-100,
        )

    def get_last_valid_date(
        self, outcome_info
    ):  # TODO: Should be standard in CensorCollate
        """Gets the last_valid_date for the outcome_info (either censor or event_final_date - negative_censor)"""
        if math.isnan(outcome_info["censor"]):
            return outcome_info["event_final_date"] - self.negative_censor
        else:
            return outcome_info["censor"]


class ScreenGridCensorCollate(GridCensorCollate):
    def get_last_valid_date(self, outcome_info):
        """add 18 months to last valid date (for grid tokens)"""
        if math.isnan(outcome_info["censor"]):
            return outcome_info["event_final_date"] - self.negative_censor
        else:
            return outcome_info["censor"] + ONE_YEAR_ABSPOS * 1.5


class ParentGridCollate(GridCollate):
    family_type = {"Child": 1, "Father": 2, "Mother": 3}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Make better
        triple_max_len = self.max_seq_len * 3
        self.len_buckets = [
            128 * (2**i) for i in range((triple_max_len // 128).bit_length())
        ]
        if triple_max_len not in self.len_buckets:
            self.len_buckets.append(triple_max_len)

    def main(self, batch: Dict[str, List]):
        # Initialize keys
        data_keys = self.data_keys(batch["data"][0]["Child"]) + ["family_type"]
        output = {key: [] for key in data_keys + ["sequence_lens"]}

        # Standard __call__ with new process_person
        for family, outcome in zip(batch["data"], batch["outcome_info"]):
            family_seq = []
            last_valid_date = self.get_last_valid_date(outcome)
            for typ, person in family.items():
                truncate_length = self.truncate_length
                if typ == "Child":
                    truncate_length -= get_n_grid_tokens(
                        last_valid_date, person["abspos"][0], self.grid_interval
                    )
                indv = self.process_person(person, truncate_length)
                if typ == "Child":
                    indv = self.add_grid_tokens_to_person(indv, last_valid_date)
                indv["family_type"] = [self.family_type[typ]] * len(indv["abspos"])
                family_seq.append(indv)

            combined = combine_and_sort_family_seq(family_seq)

            typ1_accumulated_event_lens = [
                total
                for typ, total in zip(
                    combined["family_type"],
                    itertools.accumulate(combined["event_lens"]),
                )
                if typ == 1
            ]
            combined["sequence_lens"] = typ1_accumulated_event_lens[-1]

            if self.segment:
                combined["segment"] = list(range(1, len(combined["event"]) + 1))

            # Finally flatten and expand
            for key in data_keys:
                if key == "event":
                    combined[key] = sum(combined[key], [])
                else:
                    combined[key] = self.expand(combined[key], combined["event_lens"])
                combined[key] = combined[key][: combined["sequence_lens"]]

            # Add to output
            for key, v in combined.items():
                if key in output:
                    output[key].append(v)

        # Apply padding
        self.max_seq_len *= 3  # TODO: Make better
        for key in data_keys:
            output[key] = self._pad(output[key])
        self.max_seq_len //= 3  # TODO: Make better

        output["sequence_lens"] = torch.as_tensor(output["sequence_lens"])

        return output

    def process_person(self, person, truncate_length: int = None):
        """Handles all person-specific processsing"""
        output = {}
        # Start with events
        person_seq, person_event_lens, event_border = self._flatten(
            person["event"], truncate_length=truncate_length
        )

        # Add rest of keys
        for key in person:
            truncated_seq = self._truncate(person[key], event_border)
            output[key] = truncated_seq

        output["event_lens"] = person_event_lens
        return output


class ParentMaskGridCollate(MaskCollate, ParentGridCollate):
    """MLM for GridCollate"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        GridCollate.__init__(self, *args, **kwargs)
        MaskCollate.__init__(self, *args, **kwargs)


class ParentGridCensorCollate(GridCensorCollate, ParentGridCollate):

    def __init__(self, *args, **kwargs):
        GridCensorCollate.__init__(self, *args, **kwargs)
        ParentGridCollate.__init__(self, *args, **kwargs)

    def _censor_data(self, data, outcome_info):
        for family, outcome in zip(data, outcome_info):
            for typ, person in family.items():
                censored_person = censor_person(
                    person,
                    outcome,
                    background=self.bg_events,
                    negative_censor=self.negative_censor,
                )
                family[typ] = censored_person  # Updates data in-place
        return data
