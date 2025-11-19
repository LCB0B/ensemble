""" Implements all collate functionality """

from typing import Dict, List, Tuple, Union, Literal

import torch

from src.collate_utils_new import (
    calc_age_from_abspos,
    find_censor_idx,
    censor_data,
    censor_person,
    combine_and_sort_family_seq,
)


class Collate:
    """Implements the core collate functions"""

    def __init__(self, max_seq_len=8192, background_length=0, **kwargs):
        self.max_seq_len = max_seq_len
        self.truncate_length = max_seq_len - background_length
        self.background_length = background_length
        self.bg_events = int(background_length > 0)
        self.data_keys = ["event", "age", "abspos", "segment"]

    def __call__(self, batch: Dict[str, List]) -> dict:
        """Input: List of Batch, Outcomes"""
        batch = self.pre_main(batch)

        output = self.main(batch)
        output = self.post_main(output, batch)
        return output

    def pre_main(self, batch):
        """Pre-processes batch before going into main"""
        return batch

    def post_main(self, output, batch=None):
        """Post-processes output"""
        return output

    @staticmethod
    def gather_person(batch: Dict, idx: int):
        """Gathers a Dict[key, val[idx]]"""
        return {key: val[idx] for key, val in batch.items()}

    def main(self, batch: Dict[str, List[dict]]) -> dict:
        """The main processing function"""
        output = {key: [] for key in self.data_keys}

        for i in range(len(batch["data"])):
            person = self.gather_person(batch, i)
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

        trunc_idx = self.find_truncation_idx(
            person["event_lengths"],
            truncate_length=(
                truncate_length if truncate_length is not None else self.truncate_length
            ),
        )

        trunc_event_lengths = self.truncate(person["event_lengths"], trunc_idx)
        trunc_event_lengths = torch.tensor(trunc_event_lengths, dtype=torch.int32)
        event_idxs = torch.repeat_interleave(trunc_event_lengths)

        person["data"]["segment"] = list(range(1, len(trunc_event_lengths) + 1))
        for key in self.data_keys:
            trunc_seq = self.truncate(person["data"][key], trunc_idx)
            if key == "event":
                seq = self.flatten(trunc_seq)
            else:
                seq = self.expand(trunc_seq, event_idxs)
            output[key] = seq
        return output

    def find_truncation_idx(
        self, event_lengths: List[int], truncate_length: int
    ) -> Union[None, int]:
        """Find the trucation index (returns None if not present)"""
        if (
            len(event_lengths) < truncate_length  # Small optimization
            and sum(event_lengths) < truncate_length
        ):
            return None

        n = 0
        for i, count in enumerate(reversed(event_lengths)):
            n += count
            if n > truncate_length:
                break
        else:
            return None
        return -i

    def truncate(self, seq: list, idx: int) -> list:
        """Truncates `seq` based on `idx`, keeping background if present"""
        if idx is None:
            return seq
        return seq[: self.bg_events] + seq[idx:]

    @staticmethod
    def flatten(seq: List[list]) -> list:
        """Flattens `seq`"""
        return [e for sublist in seq for e in sublist]

    @staticmethod
    def expand(seq: list, event_idxs: torch.Tensor) -> torch.Tensor:
        """Expands `seq` using indexing via. `event_idxs`"""
        dtype = torch.int32 if isinstance(seq[0], int) else torch.float32
        seq = torch.as_tensor(seq, dtype=dtype)
        return seq[event_idxs]

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


class AutoregressiveCollate(Collate):
    """Standard collate with Autoregressive shift"""

    def post_main(self, output, batch=None):
        output["target"] = output["event"][:, 1:]
        output["target"] = output["target"].masked_fill(output["target"] == 0, -100)
        for feature in ["event", "abspos", "age", "segment"]:
            output[feature] = output[feature][:, :-1]
        return super().post_main(output, batch)


class CensorCollate(Collate):
    """Standard collate with censoring"""

    def pre_main(self, batch):
        batch = self._censor_data(batch)
        return super().pre_main(batch)

    def post_main(self, output, batch):
        output = self._add_outcome_info(output, batch["outcome_info"])
        return super().post_main(output, batch)

    def _add_outcome_info(self, output: dict, outcome_info: List[dict]) -> Dict:
        output["person_id"] = [out["person_id"] for out in outcome_info]
        return output

    def _censor_data(self, batch):
        for i in range(len(batch["data"])):
            censor_idx = find_censor_idx(
                batch["data"][i]["abspos"],
                batch["outcome_info"][i]["censor"],
                background=self.bg_events,
            )
            batch["data"][i] = {
                key: value[:censor_idx] for key, value in batch["data"][i].items()
            }
            batch["event_lengths"][i] = batch["event_lengths"][i][:censor_idx]
        return batch
        # return censor_data(
        #     batch["data"], batch["outcome_info"], background=self.bg_events
        # )


class PredictCollate(Collate):
    """Standard collate with PREDICT tokens"""

    def __init__(self, *args, predict_token_id: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_token_id = predict_token_id

    def process_person(self, person, truncate_length: int = None):
        person = self.add_predict_tokens_to_person(person)
        output = super().process_person(person, truncate_length=truncate_length)
        output = self.sort_by_abspos(output)
        return output

    def add_predict_tokens_to_person(self, person):
        """Adds a PREDICT token (self.predict_token_id used) to the person for every prediction"""
        prediction_abspos = person["outcome_info"]["predict"]
        person["data"] = self.add_predict_data_to_person(
            person["data"], prediction_abspos
        )
        person["event_lengths"].extend([1] * len(prediction_abspos))
        return person

    def add_predict_data_to_person(self, person, predictions):
        """Adds predict-related features to person"""
        person["event"].extend([[self.predict_token_id] for _ in predictions])
        person["abspos"].extend(predictions)
        person["age"].extend(
            [calc_age_from_abspos(person["abspos"][0], a) for a in predictions]
        )
        return person

    @staticmethod
    def sort_by_abspos(output: Dict[str, list]):
        """Sort all keys in output by the `abspos`"""
        sorted_idxs = sorted(
            range(len(output["abspos"])), key=lambda i: output["abspos"][i]
        )

        return {key: [value[i] for i in sorted_idxs] for key, value in output.items()}


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

    def pre_main(self, batch):
        batch = super().pre_main(batch)
        for family, outcome in zip(batch["data"], batch["outcome_info"]):
            if "relations" in outcome:
                family["length_dict"] = {
                    val["relations"]: val["adjusted_lengths"]
                    for val in outcome["relations"]
                }
        return batch

    def process_person(self, family, truncate_length=None):
        family_seq = []
        length_dict = family.pop("length_dict", None)
        for typ, person in family.items():
            if typ not in self.process_set:
                continue
            truncate_length = (
                length_dict[typ] - self.background_length
                if (len(self.feature_set) > 1)
                and (length_dict[typ] <= self.truncate_length)
                else self.truncate_length
            )
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


class AutoregressivePredictCollate(PredictCollate, AutoregressiveCollate):
    """Autoregressive with Predict tokens"""

    pass


class AutoregressiveCensorPredictCollate(CensorCollate, AutoregressivePredictCollate):
    pass


class PredictCensorCollate(CensorCollate, PredictCollate):
    """Standard collate with censoring"""

    def __init__(
        self,
        *args,
        prediction_windows: list,
        padding_side: Literal["left", "right"] = None,
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
        output["target"] = self._adjust_targets([out["target"] for out in outcome_info])
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


class FamilyPredictCensorCollate(FamilyCollate, PredictCensorCollate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "Child" not in self.process_set:
            self.process_set.append("Child")

    def add_predict_info(self, batch):
        for family, outcome in zip(batch["data"], batch["outcome_info"]):
            family["Child"]["predict"] = outcome["predict"]

        return batch

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


class FamilyCensorCollate(FamilyCollate, CensorCollate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "Child" not in self.process_set:
            self.process_set.append("Child")

    def _censor_data(self, batch):
        """Censors each person in data using the info in each outcome_info"""
        censored_data = []
        for family, outcome in zip(batch["data"], batch["outcome_info"], strict=True):
            for typ, person in family.items():
                family[typ] = censor_person(person, outcome, background=self.bg_events)
            censored_data.append(family)
        return censored_data


class FamilyCensorAutoregressivePredictCollate(
    FamilyCensorCollate, AutoregressivePredictCollate
):
    def add_predict_info(self, batch):
        for family, outcome in zip(batch["data"], batch["outcome_info"]):
            family["Child"]["predict"] = outcome["predict"]

        return batch

    def process_child(self, person, truncate_length):
        return AutoregressivePredictCollate.process_person(
            self, person, truncate_length
        )

    def process_parent(self, person, truncate_length):
        return Collate.process_person(self, person, truncate_length)

    def add_predict_features_to_person(self, person, predictions):
        person = super().add_predict_features_to_person(person, predictions)
        person["family_type"].extend([person["family_type"][0]] * len(predictions))
        return person
