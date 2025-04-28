""" Implements all collate functionality """

import bisect
import itertools
from typing import List, Dict, Any, Tuple, Optional, Union
import torch
from src.utils import get_max_special_token_value, mask_inputs


class Collate:
    """Implements the core collate functions"""

    def __init__(self, truncate_length=512, background_length=0, segment=False):
        self.truncate_length = truncate_length
        self.background_length = background_length
        self.max_seq_len = truncate_length + background_length
        self.bg_events = int(self.background_length > 0)
        self.segment = segment

    def __call__(self, batch: List[dict]) -> dict:
        data_keys = [key for key in batch[0]] + ["segment"] * int(self.segment)
        output = {
            key: []
            for key in data_keys + ["last_data_idx", "event_lens", "event_borders"]
        }

        for person in batch:
            # First start with events to get required stuff for the others
            person_seq, person_event_lens, event_border = self._flatten(
                person.pop("event")
            )
            output["event"].append(person_seq)

            # Add additional info
            output["last_data_idx"].append(len(person_seq))

            if self.segment:
                person["segment"] = list(range(1, len(person_event_lens) + 1))

            for key in person:
                truncated_seq = self._truncate(person[key], event_border)
                expanded_seq = self.expand(truncated_seq, person_event_lens)
                output[key].append(expanded_seq)

            # Only needed for other collates
            output["event_borders"].append(event_border)
            output["event_lens"].append(person_event_lens)

        # Pad all person keys
        for key in data_keys:
            output[key] = self._pad(output[key])
        output["last_data_idx"] = torch.as_tensor(output["last_data_idx"])

        return output

    def _truncate(self, seq, event_border: int):
        background = [] if event_border is None else seq[: self.bg_events]
        return background + seq[event_border:]

    @staticmethod
    def expand(seq: list, repeats: list) -> torch.Tensor:
        """Repeats seq[i] repeats[i] times"""
        return torch.repeat_interleave(torch.as_tensor(seq), torch.as_tensor(repeats))

    def _flatten(self, events: List[List[int]]):
        """Flattens events and (optional) truncates, returning flatten_seq and the last event idx"""
        person_seq, person_event_lens, event_border = (
            self._flatten_reverse_and_truncate(events, self.truncate_length)
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
        # TODO: This breaks with CLS token
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
        """Pads the sequence (using padding_value) to closest defined lens and converts to tensor"""
        # Initialize lens buckets
        lens = [128 * (2**i) for i in range((self.max_seq_len // 128).bit_length())]
        if self.max_seq_len not in lens:
            lens.append(self.max_seq_len)

        # Conver to tensors and get max_len and max_idx
        sequences = [torch.as_tensor(seq, dtype=dtype) for seq in sequence]
        max_len, max_idx = torch.max(
            torch.tensor([s.size(0) for s in sequences]), dim=0
        )

        # If batch does not match a predefined len
        if max_len.values not in lens:
            closest_len = lens[bisect.bisect_left(lens, max_len)]
            extra_dims = sequences[max_idx].shape[1:]
            sequences[max_idx] = torch.cat(
                (
                    sequences[max_idx],
                    torch.full((closest_len - max_len, *extra_dims), padding_value),
                )
            )
        return torch.nn.utils.rnn.pad_sequence(
            sequences,
            batch_first=True,
            padding_value=padding_value,
        )


class MaskCollate(Collate):
    """Standard collate with masking"""

    def __init__(
        self,
        vocab: dict,
        mask_prob=0.15,
        replace_prob=0.8,
        random_prob=0.1,
        truncate_length=512,
        background_length=0,
        segment=False,
    ):
        super().__init__(truncate_length, background_length, segment=segment)
        self.vocab = vocab
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob
        self.random_prob = random_prob
        self.special_token_border = get_max_special_token_value(vocab)

    def __call__(self, batch: List[Dict]) -> Dict:
        batch = super().__call__(batch)

        # Mask events and produce targets
        batch["event"], batch["target"] = mask_inputs(
            batch["event"],
            self.vocab,
            mask_prob=self.mask_prob,
            replace_prob=self.replace_prob,
            random_prob=self.random_prob,
            special_token_border=self.special_token_border,
        )
        return batch


class CensorCollate(Collate):
    """Standard collate with censoring"""

    def __init__(
        self,
        truncate_length: int,
        background_length: int,
        segment: bool,
        negative_censor: int = 0,
    ):
        super().__init__(truncate_length, background_length, segment=segment)
        self.negative_censor = negative_censor

    def __call__(self, batch: Tuple[List[Dict]]) -> Dict:
        """Input: List of Batch, Outcomes"""
        data, outcome_info = batch
        data = self._censor(data, outcome_info)

        batch = super().__call__(data)
        batch = self._add_outcome_info(batch, outcome_info)
        return batch

    def _censor(self, data: Tuple[Dict], outcome_info: Tuple[Dict]) -> List[Dict]:
        censored_data = []
        for person, outcome in zip(data, outcome_info, strict=True):
            abspos = person["abspos"]
            censor_abspos = outcome["censor"]
            if censor_abspos is None:
                censor_abspos = (
                    abspos[-1] + 1e-10  # Tiny float adjust if self.negative_censor=0
                ) - self.negative_censor
            # Since data is sorted by abspos, we can just do binary search
            last_valid_idx = bisect.bisect_left(abspos, censor_abspos)
            last_valid_idx = max(
                last_valid_idx, self.bg_events
            )  # Always keep background
            censored_person = {
                key: value[:last_valid_idx] for key, value in person.items()
            }
            censored_data.append(censored_person)
        return censored_data

    def _add_outcome_info(self, batch: dict, outcome_info: List[dict]) -> Dict:
        batch.update(
            {key: [out[key] for out in outcome_info] for key in outcome_info[0]}
        )
        batch["target"] = self._adjust_targets(batch["target"])
        return batch

    def _adjust_targets(self, targets):
        return torch.as_tensor(targets, dtype=torch.float32)


class CausalEventCollate(CensorCollate):
    """Collate with causal event mask and multi-trajectory targets"""

    def __init__(
        self,
        truncate_length: int,
        background_length: int,
        prediction_windows: list,
        segment: bool,
        negative_censor: int = 0,
    ):
        super().__init__(
            truncate_length=truncate_length,
            background_length=background_length,
            segment=segment,
            negative_censor=negative_censor,
        )
        self.prediction_windows = torch.as_tensor(prediction_windows)

    def __call__(self, batch: List[Tuple[Dict]]) -> Dict:
        data, outcome_info = batch
        data = self._censor(data, outcome_info)
        # Calls Collate.__call__
        batch = super(CensorCollate, self).__call__(data)

        batch["event_lens"] = self._pad(
            [list(itertools.accumulate(seq)) for seq in batch["event_lens"]],
            dtype=torch.int32,
            padding_value=10_000,  # Set to higher number than max_seq_len
            # to_max_len=False,
        )
        # Create targets and add outcome_info
        batch["target"] = self._create_targets(
            data, outcome_info, batch["event_borders"]
        )
        batch = self._add_outcome_info(batch, outcome_info)
        return batch

    def _create_targets(
        self,
        data: Tuple[Dict],
        outcome_info: Tuple[Dict],
        event_borders: torch.Tensor,
    ) -> List[Dict]:
        """Creates targets (k=prediction window) per person abspos"""
        targets = []
        for person, outcome, border in zip(
            data, outcome_info, event_borders, strict=True
        ):
            abspos = self._truncate(person["abspos"], border)
            # Create targets
            if outcome["outcome"]:
                valid_targets = (
                    torch.as_tensor(abspos).unsqueeze(1) + self.prediction_windows
                ) > outcome["censor"]
            else:
                valid_targets = torch.zeros(len(abspos), len(self.prediction_windows))
            targets.append(valid_targets)
        return targets

    def _adjust_targets(self, targets):
        return self._pad(
            targets,
            dtype=torch.float32,
            padding_value=-100,
        )

    def _truncate(self, seq, event_border: int):
        if event_border is None:
            return seq

        bg, trunc_seq = seq[: self.bg_events], seq[event_border:]
        if isinstance(seq, list):
            return bg + trunc_seq
        else:
            return torch.cat((bg, trunc_seq))


class ParentCausalEventCollate(CausalEventCollate):
    # Censor
    # Truncate
    # Get event mask
    # combine

    def __call__(self, batch: Tuple[List[Dict]]) -> Dict:
        data, outcome_info, parents = batch
        # TODO: Move this?
        data, targets, parents = self._censor_and_create_targets(
            data, outcome_info, parents
        )

        output = {}
        data_keys = [key for key in data[0]]

        for add_key in data_keys + [
            "last_data_idx",
            "event_lens",
            "family_type",
            "target",
        ]:
            output[add_key] = []
        if self.segment:
            output["segment"] = []

        for person, target, family in zip(data, targets, parents):
            event_abspos = []
            event_lens = []
            family_seq = []
            indv = {}
            combined = {}
            # First start with events to get required stuff for the others
            person_seq, person_event_lens, event_border = self._flatten(
                person.pop("event")
            )
            indv["event"] = person_seq
            indv["family_type"] = [1] * len(person_seq)  # Person is type=1

            feature_keys = [key for key in data_keys if key != "event"]
            for key in feature_keys:
                truncated_seq = self._truncate(person[key], event_border)
                expanded_seq = self.expand(truncated_seq, person_event_lens)
                indv[key] = expanded_seq

            # Add only target for individual # TODO: Needed?
            combined["target"] = self._truncate(target, event_border)

            # Add rest of data
            event_abspos.extend(self._truncate(person["abspos"], event_border))
            event_lens.extend(person_event_lens)

            family_seq.append(indv)

            # Add parent information to family_seq
            for parent in family:
                typ, parent = parent.popitem()
                parent_data = {}
                parent_seq, parent_event_lens, parent_border = self._flatten(
                    parent.pop("event")
                )
                parent_data["event"] = parent_seq
                # Add rest of data
                parent_data["family_type"] = [int(typ == "Mother") + 2] * len(
                    parent_seq
                )  # Father is type=2, Mother is type=3
                event_abspos.extend(self._truncate(parent["abspos"], parent_border))
                event_lens.extend(parent_event_lens)

                for key in feature_keys:
                    truncated_seq = self._truncate(parent[key], parent_border)
                    expanded_seq = self.expand(truncated_seq, parent_event_lens)
                    parent_data[key] = expanded_seq

                family_seq.append(parent_data)

            # Combine members in family seqs
            combined_abspos = [
                abspos for person in family_seq for abspos in person["abspos"]
            ]
            sorted_indices = sorted(
                range(len(combined_abspos)), key=lambda i: combined_abspos[i]
            )
            pre_expand_sorted_indices = sorted(
                range(len(event_abspos)), key=lambda i: event_abspos[i]
            )
            for key in family_seq[0]:
                new_seq = []
                for person in family_seq:
                    new_seq.extend(person[key])
                new_seq = torch.as_tensor([new_seq[i] for i in sorted_indices])
                combined[key] = new_seq

            combined["last_data_idx"] = (combined["family_type"] == 1).nonzero()[
                -1, 0
            ] + 1
            sorted_event_lens = [event_lens[i] for i in pre_expand_sorted_indices]
            new_person_event_lens = []
            i = 0
            for event_len in sorted_event_lens:
                if combined["family_type"][i] == 1:
                    i += event_len
                    new_person_event_lens.append(i)
                else:
                    i += event_len
            combined["event_lens"] = new_person_event_lens

            if self.segment:
                combined["segment"] = self.expand(
                    list(range(1, len(event_lens) + 1)), sorted_event_lens
                )

            for key, v in combined.items():
                output[key].append(v)

        # Reset max_seq_len to actual seq_len¨
        self.max_seq_len *= 3  # TODO:

        for key in [
            "event",
            "age",
            "abspos",
            "segment",
            "family_type",
        ]:
            output[key] = self._pad(output[key])
        output["target"] = self._adjust_targets(output["target"])
        output["last_data_idx"] = torch.as_tensor(output["last_data_idx"])
        output["event_lens"] = self._pad(output["event_lens"], padding_value=10_000)

        self.max_seq_len //= 3  # TODO:
        return output

    def _censor_and_create_targets(
        self, data: Tuple[Dict], outcome_info: Tuple[Dict], parents: Tuple[Dict]
    ) -> List[Dict]:
        censored_data, targets, censored_family = [], [], []
        for person, outcome, parents in zip(data, outcome_info, parents, strict=True):
            family = []

            censor_abspos = outcome["censor"]
            if censor_abspos is None:
                censor_abspos = (
                    person["abspos"][-1] - self.negative_censor
                )  # Adjust censoring

            # For the individual
            last_valid_idx = max(
                bisect.bisect_left(person["abspos"], censor_abspos), self.bg_events
            )
            valid_person = {
                key: value[:last_valid_idx] for key, value in person.items()
            }
            # Create targets
            if outcome["outcome"]:
                valid_targets = (
                    torch.tensor(valid_person["abspos"]).unsqueeze(1)
                    + self.prediction_windows
                ) > censor_abspos
            else:
                valid_targets = torch.zeros(
                    len(valid_person["abspos"]), len(self.prediction_windows)
                )

            # For the parents
            for typ, parent in parents.items():
                last_valid_idx = max(
                    bisect.bisect_left(parent["abspos"], censor_abspos), self.bg_events
                )
                valid_parent = {
                    key: value[:last_valid_idx] for key, value in parent.items()
                }
                family.append({typ: valid_parent})

            censored_data.append(valid_person)
            targets.append(valid_targets)
            censored_family.append(family)

        return censored_data, targets, censored_family
