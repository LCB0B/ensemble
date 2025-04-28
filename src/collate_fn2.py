""" Implements all collate functionality """

import bisect
import itertools
from typing import List, Dict, Tuple, Union
import torch
import math
from src.utils import get_max_special_token_value, mask_inputs, mask_inputs_for_next_token_prediction
from torch.nn.attention.flex_attention import  create_block_mask


import pdb
ONE_YEAR_ABSPOS = 365.25 * 24


class Collate:
    """Implements the core collate functions"""

    def __init__(self, truncate_length=512, background_length=0, segment=False):
        self.truncate_length = truncate_length
        self.background_length = background_length
        self.max_seq_len = truncate_length + background_length
        self.bg_events = int(self.background_length > 0)
        self.segment = segment

        self.len_buckets = [
            128 * (2**i) for i in range((self.max_seq_len // 128).bit_length())
        ]
        if self.max_seq_len not in self.len_buckets:
            self.len_buckets.append(self.max_seq_len)

    def __call__(self, batch: List[dict]) -> dict:
        """
        Processes a batch of data, skipping items with invalid 'abspos'.
        """
        # --- Check if the input batch list itself is empty ---
        if not batch:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!! 🔥🔥 WARNING: CollateFn received EMPTY batch list !!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # Return empty dict, might cause errors downstream if not handled
            return {}
        # --- Check if first item is a dict to determine keys ---
        if not isinstance(batch[0], dict):
             print(f"🔥🔥 WARNING: First item in batch is not a dict. Type: {type(batch[0])}. Cannot process. Returning empty batch.")
             return {}

        # --- Initialize output ---
        # Get keys from the first item, assuming structure is consistent
        # Ensure 'person_id' is handled gracefully if it's sometimes missing from keys
        base_keys = [key for key in batch[0] if key != 'person_id'] # Exclude person_id if it's just metadata here
        data_keys = base_keys + ["segment"] * int(self.segment)
        output = {key: [] for key in data_keys + ["sequence_lens"]} # Ensure seq_lens key is initialized
        valid_items_processed = 0

        # --- Loop through items, check validity, and process ---
        for item_idx, person in enumerate(batch):

            # --- VALIDITY CHECK ---
            valid_item = True
            if not isinstance(person, dict):
                print(f"🛑 WARNING: Item at index {item_idx} is not a dict. Type: {type(person)}. Skipping.")
                valid_item = False
            else:
                abspos_val = person.get("abspos")
                # Checks for None, not a list, or empty list []
                if not isinstance(abspos_val, list) or not abspos_val:
                    # Try to get an ID if available, otherwise use index
                    person_id = person.get("person_id", f"NO_ID_at_batch_index_{item_idx}")
                    reason = "missing" if abspos_val is None else ("not a list" if not isinstance(abspos_val, list) else "empty")
                    # Use a distinct marker for easy searching
                    print(f"@@@ SKIP INFO @@@ Skipping item for ID {person_id} (batch index {item_idx}) due to {reason} 'abspos'. Value: {repr(abspos_val)}") # Use repr for clarity
                    valid_item = False
            # --- END VALIDITY CHECK ---

            if not valid_item:
                continue # Skip to the next person in the batch

            # --- Process valid items ---
            try:
                # Use copy() because process_person pops 'event'
                # Ensure process_person can handle the actual valid data structure
                indv = self.process_person(person.copy())

                # Append processed data to output lists
                for key, v in indv.items():
                     # Only add keys that were expected based on the first item (or added like segment)
                     if key in output:
                          output[key].append(v)
                     # else: # Optional: Warn about unexpected keys from process_person
                     #     print(f"Warning: Unexpected key '{key}' from process_person ignored.")


                # Add sequence length - Use .get on the result 'indv' for safety
                output["sequence_lens"].append(len(indv.get("event", [])))
                valid_items_processed += 1

            except KeyError as ke:
                 # Catch missing keys during process_person (e.g., if 'event' was missing even after check)
                 person_id = person.get("person_id", f"NO_ID_at_batch_index_{item_idx}")
                 print(f"🟥 ERROR (KeyError) processing item for ID {person_id} (batch index {item_idx}): Missing key {ke}. Skipping.")
                 continue
            except Exception as e:
                 # Catch other unexpected errors during individual processing
                 person_id = person.get("person_id", f"NO_ID_at_batch_index_{item_idx}")
                 print(f"🟥 ERROR (General) processing item for ID {person_id} (batch index {item_idx}): {type(e).__name__} - {e}. Skipping.")
                 # import traceback; traceback.print_exc() # Uncomment for full traceback
                 continue


        # --- Handle case where ALL items in the batch were skipped ---
        if valid_items_processed == 0:
             print("🔥🔥 WARNING: Entire batch skipped due to invalid items. Returning empty batch structure.")
             return {}

        # --- Pad the collected valid items ---
        # Check sequence_lens list exists and has data before padding others
        if "sequence_lens" not in output or not output["sequence_lens"]:
             print("🔥🔥 WARNING: 'sequence_lens' key missing or empty after processing loop. Returning empty batch.")
             return {}

        keys_to_pad = list(output.keys()) # Get keys present in the output dict

        for key in keys_to_pad:
             # Skip sequence_lens for now, handle it separately
             if key == "sequence_lens":
                  continue
             if key not in output or not output[key]: # Double check list has data
                   print(f"🟡 INFO: Output key '{key}' is missing or empty before padding, creating empty tensor.")
                   # Create an empty tensor with appropriate dtype if known, otherwise default
                   output[key] = torch.empty((0,), dtype=torch.int64) # Adjust shape/dtype as needed
                   continue # Skip padding call

             try:
                   # Determine padding value (usually 0, but might differ)
                   padding_value = 0
                   # Get dtype from first element if possible, default otherwise
                   dtype = output[key][0].dtype if isinstance(output[key][0], torch.Tensor) else None

                   output[key] = self._pad(output[key], dtype=dtype, padding_value=padding_value)
             except Exception as e:
                   print(f"🟥 ERROR during padding key '{key}': {e}. Returning empty batch.")
                   return {}


        # Convert sequence lengths to tensor after all other padding is done
        try:
            output["sequence_lens"] = torch.as_tensor(output["sequence_lens"])
        except Exception as e:
            print(f"🟥 ERROR converting sequence_lens to tensor: {e}. Returning empty batch.")
            return {}


        return output

    def process_person(self, person):
        """Handles all person-specific processsing"""
        output = {}
        # Start with events
        person_seq, person_event_lens, event_border = self._flatten(person.pop("event"))
        output["event"] = person_seq

        if self.segment:
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
        """Pads the sequence (using padding_value) to closest bucket in self.len_buckets and converts to tensor"""
        # Conver to tensors and get max_len and max_idx
        sequences = [torch.as_tensor(seq, dtype=dtype) for seq in sequence]
        max_len, max_idx = torch.max(
            torch.tensor([s.size(0) for s in sequences]), dim=0
        )

        # If batch does not match a predefined len bucket
        if max_len.values not in self.len_buckets:
            closest_len = self.len_buckets[
                bisect.bisect_left(self.len_buckets, max_len)
            ]
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


class CausalCollate(Collate):
    """Collate function for next-token prediction with causal masking."""

    def __init__(
        self,
        vocab: dict,
        truncate_length=512,
        background_length=0,
        segment=False,
    ):
        """
        Initializes the CausalCollate class.

        Args:
            vocab (dict): Vocabulary dictionary (optional, unused in this implementation).
            truncate_length (int): Maximum sequence length for truncation.
            background_length (int): Length of any additional background tokens.
            segment (bool): Whether to include segment embeddings.
        """
        super().__init__(truncate_length, background_length, segment=segment)
        self.vocab = vocab

    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Processes a batch of data for next-token prediction.

        Args:
            batch (List[Dict]): List of dictionaries containing "event" sequences.

        Returns:
            Dict: Processed batch with "event", "target", and "attn_mask".
        """
        # Process the batch using the parent Collate class
        batch = super().__call__(batch)

        # Prepare events and targets for next-token prediction
        try:
            batch["event"], batch["target"] = mask_inputs_for_next_token_prediction(batch["event"])
        except Exception as e:
            raise RuntimeError(f"Error in next-token masking: {e}")

    # # Create causal attention mask (mistake ithink see below)
    #     bs, seq_len = batch["event"].shape

    #     def causal_mask(b,h,q_idx,kv_idx):
    #         return q_idx >= kv_idx
    
    #     # Create block mask
    #     batch["attn_mask"] = create_block_mask(
    #         causal_mask,
    #         bs,
    #         None,
    #         seq_len,
    #         seq_len,
    #         _compile=True,
    #     )

        return batch

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
        if __debug__:
            pdb.set_trace()
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
            censored_person = self._censor_person(person, outcome["censor"])
            censored_data.append(censored_person)
        return censored_data

    def _censor_person(self, person, censor_abspos):
        abspos = person["abspos"]
        if censor_abspos is None:
            censor_abspos = (
                abspos[-1] + 1e-10  # Tiny float adjust if self.negative_censor=0
            ) - self.negative_censor
        # Since data is sorted by abspos, we can just do binary search
        last_valid_idx = bisect.bisect_left(abspos, censor_abspos)
        last_valid_idx = max(last_valid_idx, self.bg_events)  # Always keep background
        return {key: value[:last_valid_idx] for key, value in person.items()}

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
        max_abspos: float,
        negative_censor: int = 0,
    ):
        super().__init__(
            truncate_length=truncate_length,
            background_length=background_length,
            segment=segment,
            negative_censor=negative_censor,
        )
        self.max_abspos = max_abspos + 1e-9
        self.prediction_windows = torch.as_tensor(prediction_windows)

    def __call__(self, batch: List[Tuple[Dict]]) -> Dict:
        batch = super().__call__(batch)

        batch["first_abspos"] = torch.as_tensor(batch["first_abspos"])
        batch["acc_event_lens"] = self._pad(
            batch["acc_event_lens"],
            dtype=torch.int32,
            padding_value=1e4,  # Set to higher number than max_seq_len
        )
        return batch

    def add_grid_tokens_to_person(self, person):
        """Adds a GRID token ([MASK] used) to the person every 0.5 year"""
        abspos_range = torch.arange(
            person["abspos"][0], self.max_abspos, ONE_YEAR_ABSPOS * 0.5
        )
        person["event"].extend([[4] for _ in abspos_range])
        person["abspos"].extend(abspos_range.tolist())
        person["age"].extend([y / 2 for y in range(len(abspos_range))])
        person["event_lens"].extend([1] * len(abspos_range))

        sorted_idxs = sorted(
            range(len(person["abspos"])), key=lambda i: person["abspos"][i]
        )

        return {key: [value[i] for i in sorted_idxs] for key, value in person.items()}

    def process_person(self, person):
        """Handles all person-specific processsing"""
        output = {}
        # Start with events
        n_grid_tokens = math.ceil(
            (self.max_abspos - person["abspos"][0]) / (ONE_YEAR_ABSPOS * 0.5)
        )
        person_seq, person_event_lens, event_border = self._flatten(
            person["event"], truncate_length=self.truncate_length - n_grid_tokens
        )
        output["event_lens"] = person_event_lens
        # output["event"] = person_seq

        # Add rest of keys
        for key in person:
            truncated_seq = self._truncate(person[key], event_border)
            # expanded_seq = self.expand(truncated_seq, person_event_lens) # Skip expanding
            output[key] = truncated_seq

        output = self.add_grid_tokens_to_person(output)

        if self.segment:
            output["segment"] = list(range(1, len(output["event_lens"]) + 1))

        # Expand
        for key in output:
            if key == "event_lens":
                continue
            elif key == "event":
                output[key] = sum(output[key], [])
            else:
                output[key] = self.expand(output[key], output["event_lens"])

        output["acc_event_lens"] = torch.as_tensor(
            list(itertools.accumulate(output["event_lens"]))
        )
        output["first_abspos"] = output["abspos"][0]
        return output

    def _add_outcome_info(self, batch: dict, outcome_info: List[dict]) -> Dict:
        outcome_info = self._create_targets(batch, outcome_info)
        return super()._add_outcome_info(batch, outcome_info)

    def _create_targets(
        self,
        batch: Dict,
        outcome_info: Tuple[Dict],
    ) -> List[Dict]:
        """Creates targets (k=prediction window) per person event"""
        for i, outcome in enumerate(outcome_info):
            abspos = torch.gather(batch["abspos"][i], 0, batch["acc_event_lens"][i] - 1)
            outcome["target"] = self.construct_targets_grid(abspos, outcome)
        return outcome_info

    def construct_targets_grid(self, abspos, outcome):
        """Creates targets (k=prediction window) per abspos"""
        grid = torch.arange(abspos[0], abspos[-1], ONE_YEAR_ABSPOS * 0.5)
        if outcome["outcome"]:
            return (grid.unsqueeze(1) + self.prediction_windows) > outcome["censor"]
        else:
            return torch.zeros(len(grid), len(self.prediction_windows))

    def _adjust_targets(self, targets):
        return self._pad(
            targets,
            dtype=torch.float32,
            padding_value=-100,
        )


class ParentCausalEventCollate(CausalEventCollate):
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

    def _censor(self, family, outcome_info):
        censor_abspos = outcome_info["censor"]
        for typ, person in family.items():
            censored_person = self._censor_person(person, censor_abspos)
            family[typ] = censored_person
        return family

    def process_person(self, person, truncate_length: int = None):
        """Handles all person-specific processsing"""
        output = {}
        # Start with events (no person.pop)
        person_seq, person_event_lens, event_border = self._flatten(
            person["event"], truncate_length=truncate_length
        )

        # Add rest of keys
        for key in person:
            truncated_seq = self._truncate(person[key], event_border)
            # expanded_seq = self.expand(truncated_seq, person_event_lens) # Skip expanding
            output[key] = truncated_seq

        output["event_lens"] = person_event_lens
        return output

    @staticmethod
    def get_abspos_sorted_indices(family_seq: List[dict]):
        combined_abspos = [
            abspos for person in family_seq for abspos in person["abspos"]
        ]
        return sorted(range(len(combined_abspos)), key=lambda i: combined_abspos[i])

    @staticmethod
    def combine_and_sort_family_seq(family_seq: List[dict], sorted_indices: list):
        """Combine family_seq and re-sort the data"""
        combined = {key: [] for key in family_seq[0]}
        for person in family_seq:
            for key, v in person.items():
                combined[key].extend(v)
        return {key: [v[i] for i in sorted_indices] for key, v in combined.items()}

    def __call__(self, batch: Tuple[List[Dict]]) -> Dict:
        data, outcome_info = batch
        data_keys = (
            [key for key in data[0]["Child"]]
            + ["segment"] * int(self.segment)
            + ["family_type"]
        )
        output = {
            key: []
            for key in data_keys
            + ["sequence_lens", "acc_event_lens", "target", "first_abspos"]
        }

        for family, outcome in zip(data, outcome_info):
            family = self._censor(family, outcome)
            family_seq = []
            for typ, person in family.items():
                if typ == "Child":
                    n_grid_tokens = math.ceil(
                        (self.max_abspos - person["abspos"][0])
                        / (ONE_YEAR_ABSPOS * 0.5)
                    )
                    truncate_length = self.truncate_length - n_grid_tokens
                else:
                    truncate_length = self.truncate_length

                indv = self.process_person(person, truncate_length=truncate_length)
                if typ == "Child":
                    indv = self.add_grid_tokens_to_person(indv)
                    targets = self.construct_targets_grid(indv["abspos"], outcome)
                    first_abspos = indv["abspos"][0]
                indv["family_type"] = [self.family_type[typ]] * len(indv["abspos"])
                family_seq.append(indv)
            # Get sorted indices based on abspos
            sorted_indices = self.get_abspos_sorted_indices(family_seq)

            # Combine and re-sort the data
            combined = self.combine_and_sort_family_seq(family_seq, sorted_indices)
            combined["target"] = targets
            combined["first_abspos"] = first_abspos

            # Gather accumulated event_lens for the person (typ1)
            typ1_accumulated_event_lens = [
                total
                for typ, total in zip(
                    combined["family_type"],
                    itertools.accumulate(combined["event_lens"]),
                )
                if typ == 1
            ]
            combined["acc_event_lens"] = typ1_accumulated_event_lens
            combined["sequence_lens"] = typ1_accumulated_event_lens[-1]

            if self.segment:
                combined["segment"] = list(range(1, len(combined["event"]) + 1))

            # Finally flatten and expand
            for key in data_keys:
                if key == "event":
                    combined[key] = sum(combined[key], [])
                else:
                    combined[key] = self.expand(combined[key], combined["event_lens"])

            for key, v in combined.items():
                if key in output:
                    output[key].append(v)

        # Reset max_seq_len to actual seq_len¨
        self.max_seq_len *= 3  # TODO: Make better
        for key in data_keys:
            output[key] = self._pad(output[key])
        self.max_seq_len //= 3  # TODO: Make better

        output["target"] = self._adjust_targets(output["target"])
        output["sequence_lens"] = torch.as_tensor(output["sequence_lens"])
        output["first_abspos"] = torch.as_tensor(output["first_abspos"])
        output["acc_event_lens"] = self._pad(
            output["acc_event_lens"], padding_value=10_000
        )

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
