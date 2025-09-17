import bisect
import math
from typing import Dict, List, Tuple

import torch

ONE_YEAR_ABSPOS = 365.25 * 24


# Taken from https://github.com/huggingface/transformers/blob/75f15f39a0434fe7a61385c4677f2700542a7ba6/src/transformers/data/data_collator.py#L817
def mask_inputs(
    inputs: torch.Tensor,
    vocab: dict,
    mask_prob=0.15,
    replace_prob=0.8,
    random_prob=0.1,
):
    """Masks inputs using the 80-10-10 strategy"""
    assert (replace_prob + random_prob) <= 1
    assert 0 <= mask_prob < 1
    # inputs must be pre-padded and a tensor
    targets = inputs.clone().long()
    probability_matrix = torch.full(targets.shape, mask_prob)
    special_tokens_mask = get_special_tokens_mask(targets, vocab)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    targets[~masked_indices] = -100  # Only compute loss for masked indices

    # Replace tokens with [MASK] with probability replace_prob
    indices_replaced = (
        torch.bernoulli(torch.full(targets.shape, replace_prob)).bool() & masked_indices
    )
    inputs[indices_replaced] = vocab["[MASK]"]

    # Replace tokens with random with probability random_prob
    random_prob = random_prob / (
        1 - replace_prob
    )  # Adjust probability to account for already masked tokens
    indices_random = (
        torch.bernoulli(torch.full(targets.shape, random_prob)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(vocab), targets.shape, dtype=inputs.dtype)
    inputs[indices_random] = random_words[indices_random]

    # Ignore tokens with probability 1 - replace_prob - random_prob
    return inputs, targets


def get_special_tokens_mask(inputs: torch.Tensor, vocab: dict):
    """Gets the special token mask for inputs"""
    special_tokens = []
    for k, v in vocab.items():
        if k.startswith("["):
            special_tokens.append(v)
        else:
            break

    special_token_border = max(special_tokens)
    return inputs <= special_token_border


def censor_data(
    data: Tuple[Dict], outcome_info: Tuple[Dict], background: int = 1
) -> List[Dict]:
    """Censors each person in data using the info in each outcome_info"""
    censored_data = []
    for person, outcome in zip(data, outcome_info, strict=True):
        censored_person = censor_person(person, outcome, background=background)
        censored_data.append(censored_person)
    return censored_data


def censor_person(person, outcome_info, background):
    """Removes events after the last_valid_date"""

    abspos = person["abspos"]
    censor_abspos = outcome_info["censor"]
    # Since data is sorted by abspos, we can just do binary search
    last_valid_idx = bisect.bisect_left(abspos, censor_abspos)
    last_valid_idx = max(last_valid_idx, background)  # Always keep background
    return {key: value[:last_valid_idx] for key, value in person.items()}


def calc_age_from_abspos(first_abspos, target_abspos):
    """Calculates the amount of yearas between first_abspos and target_abspos"""
    return (target_abspos - first_abspos) / ONE_YEAR_ABSPOS


def get_abspos_sorted_indices(family_seq: List[dict]):
    """Returns the sorted indices from a list of dicts with 'abspos' key"""
    combined_abspos = [abspos for person in family_seq for abspos in person["abspos"]]
    return sorted(range(len(combined_abspos)), key=lambda i: combined_abspos[i])


def combine_and_sort_family_seq(family_seq: List[dict]):
    """Combines family_seq and re-sort the data based on abspos"""
    sorted_indices = get_abspos_sorted_indices(family_seq)
    combined = {key: [] for key in family_seq[0]}
    for person in family_seq:
        for key, v in person.items():
            combined[key].extend(v)
    return {key: [v[i] for i in sorted_indices] for key, v in combined.items()}
