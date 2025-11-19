import bisect
from typing import List, Dict, Tuple

ONE_YEAR_ABSPOS = 365.25 * 24


def find_censor_idx(abspos, censor, background):
    """Finds censoring idx"""
    last_valid_idx = bisect.bisect_left(abspos, censor)
    last_valid_idx = max(last_valid_idx, background)  # Always keep background
    return last_valid_idx


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
