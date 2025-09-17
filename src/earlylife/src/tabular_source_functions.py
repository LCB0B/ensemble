# %%
import os

os.environ["POLARS_MAX_THREADS"] = "8"
os.environ["RAYON_NUM_THREADS"] = "8"

import functools  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from typing import Any, Callable  # noqa: E402

import polars as pl  # noqa: E402
from loguru import logger as default_logger  # noqa: E402

from src.earlylife.src.paths import (  # noqa: E402
    FPATH,
    check_and_copy_file_or_dir,
    copy_file_or_dir,
)
from src.earlylife.src.tabular_preprocessing import (  # noqa: E402
    calculate_mean_grade_pool_infrequent,
    calculate_means_modes_sums,
    calculate_mode_by_personid,
    calculate_weighted_mean_grade_pool_infrequent,
    collect_filtered_parquet,
    compute_weighted_past_average,
    filter_and_flag_events,
    filter_and_flag_events_with_binary,
    filter_and_flag_events_with_end_date,
    group_values_multiple,
)


def create_timing_decorator(log: Any = default_logger) -> Callable:
    """
    Create a timing decorator that logs the start time, elapsed time, and function name using the provided logger.

    Args:
        log (Any): A logger instance (e.g., from loguru).

    Returns:
        Callable: A decorator that logs timing information.
    """

    def timing_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            log.info(
                f"Starting {func.__name__} at {time.strftime('%H:%M:%S', time.localtime(start_time))}"
            )
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            log.info(f"Finished {func.__name__} in {elapsed_time:.2f} seconds")
            return result

        return wrapper

    return timing_decorator


def preprocess_amrun(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    disco_truncation_level: int,
    bra_truncation_level: int,
    previous_years: int,
    df_sample: pl.DataFrame,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the AMRUN dataset filtered by person IDs and years.

    Args:
        filename (str): Base filename (e.g., "amrun_DUMP") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        disco_truncation_level (int): Number of characters to retain for the 'disco_kode' column.
        bra_truncation_level (int): Number of characters to retain for the 'arb_hoved_bra_db07' column.
        previous_years (int): Number of previous years to consider when calculating means and modes.
        df_sample (pl.DataFrame): DataFrame containing sample data to extract person IDs per year.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """
    # Load the DataFrame filtered by person_ids
    df_amrun = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    )

    # Transform the DataFrame columns with specified truncation levels
    df_amrun = df_amrun.with_columns(
        pl.col("disco_kode")
        .cast(pl.Utf8)
        .str.slice(0, disco_truncation_level)
        .alias("disco_kode"),
        pl.col("arb_hoved_bra_db07")
        .cast(pl.Utf8)
        .str.slice(0, bra_truncation_level)
        .alias("arb_hoved_bra_db07"),
    )

    amrun_dataframes = []
    for year in years:
        result = calculate_means_modes_sums(
            df_amrun,
            person_id=df_sample.filter(pl.col("year") == year)["person_id"].to_list(),
            reference_year=year,
            mean_columns=["tilstand_grad_amr"],
            mode_columns=[
                "soc_status_kode",
                "disco_kode",
                "arb_hoved_bra_db07",
                "tilstand_kode_amr",
                "fravaer_besk_kode",
                "stoette_besk_kode",
                "udd_besk_kode",
            ],
            sum_columns=["bredt_loen_beloeb"],
            previous_years=previous_years,
        )
        result = result.with_columns(pl.lit(year).alias("year"))

        amrun_dataframes.append(result)

    all_amruns = pl.concat(amrun_dataframes)

    # Construct the output file path in the specified folder and save the DataFrame
    output_file = FPATH.DATA / folder / "amrun.parquet"
    all_amruns.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_at_risk(
    filename: str,
    person_ids: list[int],
    folder: str,
    df_sample: pl.DataFrame,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save yearly at-risk status with weighted history for each person.

    Args:
        filename (str): Base filename (e.g., "amrun_DUMP") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        df_sample (pl.DataFrame): DataFrame containing sample with 'person_id' and 'year'.
        force_copy (bool): Whether to force copy of file from opposite drive.
    """
    # Load data
    yearly_at_risk_status = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    )

    # Compute weighted past average
    df_avg = compute_weighted_past_average(yearly_at_risk_status)

    df = df_sample.join(df_avg, how="left", on=["person_id", "year"])

    # Save result
    output_file = FPATH.DATA / folder / "at_risk.parquet"
    df.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_ind(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    df_sample: pl.DataFrame,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the IND dataset filtered by person IDs and years.

    Args:
        filename (str): Base filename (e.g., "ind_DUMP") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        previous_years (int): Number of previous years to consider when calculating means and modes.
        df_sample (pl.DataFrame): DataFrame containing sample data to extract person IDs per year.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """
    # Load the DataFrame filtered by person_ids
    df_ind = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    )

    ind_dataframes = []
    for year in years:
        # Use df_sample (previously df_neet) to extract person IDs for the current year
        result = df_ind.filter(
            pl.col("person_id").is_in(
                df_sample.filter(pl.col("year") == year)["person_id"].to_list()
            )
            & (pl.col("referencetid").dt.year() == year - 1)
        )

        value_cols = [
            "dispon_13",
            "perindkialt_13",
            "erhvervsindk_13",
            "loenmv_13",
            "honny",
            "netovskud_13",
            "off_overforsel_13",
            "dagpenge_kontant_13",
            "arblhumv",
            "ovrig_dagpenge_akas_13",
            "kontanthj_13",
            "ovrig_kontanthjalp_13",
            "syg_barsel_13",
            "ovrig_overforsel_13",
            "stip",
            "korstoett",
            "korydial",
            "gron_check",
            "offpens_efterlon_13",
            "folkefortid_13",
            "qeftlon",
            "privat_pension_13",
            "formueindk_brutto",
            "resuink_13",
            "lejev_egen_bolig",
            "rentudgpr",
            "skatmvialt_13",
            "underhol",
            "assets",
        ]
        result = result.select(["person_id"] + value_cols)

        result = result.with_columns(pl.lit(year).alias("year"))

        ind_dataframes.append(result)

    all_inds = pl.concat(ind_dataframes)

    # Construct the output file path in the specified folder and save the DataFrame
    output_file = FPATH.DATA / folder / "ind.parquet"
    all_inds.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_akm(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    truncation_levels: dict[str, int],
    previous_years: int,
    df_sample: pl.DataFrame,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the AKM dataset filtered by person IDs and years.

    Args:
        filename (str): Base filename (e.g., "akm_DUMP") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        disco_truncation_level (int): Number of characters to retain for the 'disco_kode' column.
        bra_truncation_level (int): Number of characters to retain for the 'arb_hoved_bra_db07' column.
        previous_years (int): Number of previous years to consider when calculating means and modes.
        df_sample (pl.DataFrame): DataFrame containing sample data to extract person IDs per year.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """
    # Load the DataFrame filtered by person_ids
    df_akm = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    )

    # Transform the DataFrame columns with specified truncation levels
    df_akm = df_akm.with_columns(
        pl.col("disco08_alle_indk_13")
        .cast(pl.Utf8)
        .str.slice(0, truncation_levels["TRUNCATION_LEVEL_DISCO08"]),
        pl.col("disco08_loen_indk")
        .cast(pl.Utf8)
        .str.slice(0, truncation_levels["TRUNCATION_LEVEL_DISCO08"]),
        pl.col("disco08_sel_indk")
        .cast(pl.Utf8)
        .str.slice(0, truncation_levels["TRUNCATION_LEVEL_DISCO08"]),
        pl.col("disco_alle_indk_13")
        .cast(pl.Utf8)
        .str.slice(0, truncation_levels["TRUNCATION_LEVEL_DISCO"]),
        pl.col("discoloen_indk")
        .cast(pl.Utf8)
        .str.slice(0, truncation_levels["TRUNCATION_LEVEL_DISCO"]),
        pl.col("discosel_indk")
        .cast(pl.Utf8)
        .str.slice(0, truncation_levels["TRUNCATION_LEVEL_DISCO"]),
        pl.col("nace_db07_13")
        .cast(pl.Utf8)
        .str.slice(0, truncation_levels["TRUNCATION_LEVEL_NACE_DB07"]),
        pl.col("nacea_db07")
        .cast(pl.Utf8)
        .str.slice(0, truncation_levels["TRUNCATION_LEVEL_NACE_DB07"]),
        pl.col("nacei_db07")
        .cast(pl.Utf8)
        .str.slice(0, truncation_levels["TRUNCATION_LEVEL_NACE_DB07"]),
        pl.col("nace_13")
        .cast(pl.Utf8)
        .str.slice(0, truncation_levels["TRUNCATION_LEVEL_NACE_13"]),
        pl.col("nacea")
        .cast(pl.Utf8)
        .str.slice(0, truncation_levels["TRUNCATION_LEVEL_NACE_13"]),
        pl.col("nacei")
        .cast(pl.Utf8)
        .str.slice(0, truncation_levels["TRUNCATION_LEVEL_NACE_13"]),
        pl.col("branche_77")
        .cast(pl.Utf8)
        .str.slice(0, truncation_levels["TRUNCATION_LEVEL_BRANCHE77"]),
        pl.col("brchi")
        .cast(pl.Utf8)
        .str.slice(0, truncation_levels["TRUNCATION_LEVEL_BRANCHE77"]),
        pl.col("brchl")
        .cast(pl.Utf8)
        .str.slice(0, truncation_levels["TRUNCATION_LEVEL_BRANCHE77"]),
    )

    akm_dataframes = []
    for year in years:
        # Use df_sample (previously df_neet) to extract person IDs for the current year
        result = calculate_means_modes_sums(
            df_akm,
            person_id=df_sample.filter(pl.col("year") == year)["person_id"].to_list(),
            reference_year=year,
            mean_columns=None,
            mode_columns=[
                "socio_gl",
                "disco08_alle_indk_13",
                "disco08_loen_indk",
                "disco08_sel_indk",
                "disco_alle_indk_13",
                "discoloen_indk",
                "discosel_indk",
                "nace_db07_13",
                "nacea_db07",
                "nacei_db07",
                "nace_13",
                "nacea",
                "nacei",
                "socio13",
                "branche_77",
                "brchi",
                "brchl",
            ],
            previous_years=previous_years,
            date_col="referencetid",
        )
        result = result.with_columns(pl.lit(year).alias("year"))

        akm_dataframes.append(result)

    all_akms = pl.concat(akm_dataframes)

    # Construct the output file path in the specified folder and save the DataFrame
    output_file = FPATH.DATA / folder / "akm.parquet"
    all_akms.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_lpr(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the LPR dataset filtered by person IDs and years.

    Args:
        filename (str): Base filename (e.g., "lpradm_DUMP") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """
    df_lpr = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    )

    lpr_dataframes = []
    for year in years:
        previous_events = df_lpr.filter(pl.col("dato_start").dt.year() < year)
        result = group_values_multiple(
            previous_events, "person_id", ["aktionsdiagnose", "urgency", "patienttype"]
        )
        result = result.with_columns(pl.lit(year).alias("year"))
        lpr_dataframes.append(result)

    all_lprs = pl.concat(lpr_dataframes)

    output_file = FPATH.DATA / folder / "lpr.parquet"
    all_lprs.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_lmdb(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the LMDB dataset filtered by person IDs and years.

    Args:
        filename (str): Base filename (e.g., "lmdb_DUMP") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """
    df_lmdb = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    )

    lmdb_dataframes = []
    for year in years:
        previous_events = df_lmdb.filter(pl.col("eksd").dt.year() < year)
        result = group_values_multiple(previous_events, "person_id", "atc")
        result = result.with_columns(pl.lit(year).alias("year"))
        lmdb_dataframes.append(result)

    all_lmdbs = pl.concat(lmdb_dataframes)

    output_file = FPATH.DATA / folder / "lmdb.parquet"
    all_lmdbs.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_sygesik(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the LMDB dataset filtered by person IDs and years.

    Args:
        filename (str): Base filename (e.g., "lmdb_DUMP") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """
    df_sygesik = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    )

    sygesik_dataframes = []
    for year in years:
        previous_events = df_sygesik.filter(pl.col("date_col").dt.year() < year)
        result = group_values_multiple(previous_events, "person_id", "speciale")
        result = result.with_columns(pl.lit(year).alias("year"))
        sygesik_dataframes.append(result)

    all_sygesiks = pl.concat(sygesik_dataframes)

    output_file = FPATH.DATA / folder / "sygesik.parquet"
    all_sygesiks.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


# %%


def preprocess_udgk(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    min_occurrence_time: int,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the UDGK dataset filtered by person IDs and years.

    The function loads the full unfiltered DataFrame, identifies courses occurring at least
    `min_occurrence_time`, and computes mean grades. Frequent courses get their own mean columns,
    while the remaining courses are averaged into a single overall mean column.

    Args:
        filename (str): (str) Base filename (e.g., "karakter_gym_f_DUMP") without extension.
        person_ids (list[int]): (list[int]) List of person IDs for filtering the DataFrame.
        years (list[int]): (list[int]) List of years for processing the DataFrame.
        folder (str): (str) Folder name under FPATH.DATA where the output file will be saved.
        min_occurrence_time (int): (int) Minimum number of occurrences for a course to have its own column.
        force_copy (bool): (bool) Whether to force copying of the file from the opposite drive. Default is False.
    """
    # Copy if forced to
    if force_copy:
        copy_file_or_dir(FPATH.DUMP_DIR / f"{filename}.parquet")
    else:
        check_and_copy_file_or_dir(FPATH.DUMP_DIR / f"{filename}.parquet")

    # Load the full unfiltered DataFrame.
    df_udgk = pl.read_parquet(FPATH.DUMP_DIR / f"{filename}.parquet")

    # Identify frequent courses based on min_occurrence_time.
    course_counts = df_udgk.group_by("course").agg(pl.count("course").alias("count"))
    frequent_courses = course_counts.filter(pl.col("count") >= min_occurrence_time)[
        "course"
    ].to_list()

    # Filter for specified person IDs.
    relevant_events = df_udgk.filter(pl.col("person_id").is_in(person_ids))

    # Free up memory
    del df_udgk

    udgk_dataframes = []
    for year in years:
        # Filter events that occurred before the current year.
        previous_events = relevant_events.filter(pl.col("date_col").dt.year() < year)

        dataframe_person_id_year = (
            pl.DataFrame({"person_id": previous_events["person_id"].unique().to_list()})
            .cast({"person_id": int})
            .with_columns(pl.lit(year).alias("year"))
        )

        # Compute mean grade with separate columns for frequent courses.
        result_frequent_df, result_infrequent_df = calculate_mean_grade_pool_infrequent(
            previous_events,
            "person_id",
            "course",
            "grade",
            alias_prefix="hs_mean_",
            frequent_courses=frequent_courses,
        )
        modes = calculate_mode_by_personid(
            previous_events, ["instnr", "udd"], "hs_mode_"
        )

        result = (
            dataframe_person_id_year.join(
                result_frequent_df, how="left", on="person_id"
            )
            .join(result_infrequent_df, how="left", on="person_id")
            .join(modes, how="left", on="person_id")
        )
        udgk_dataframes.append(result)

    all_udgk = pl.concat(udgk_dataframes, how="diagonal_relaxed")

    output_file = FPATH.DATA / folder / "udgk.parquet"
    all_udgk.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_udfk(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    min_occurrence_time: int,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the UDFK dataset filtered by person IDs and years.

    The function loads the full unfiltered DataFrame, identifies courses occurring at least
    `min_occurrence_time`, and computes mean grades. Frequent courses get their own mean columns,
    while the remaining courses are averaged into a single overall mean column.

    Args:
        filename (str): (str) Base filename (e.g., "udfk_DUMP_AUGMENTED") without extension.
        person_ids (list[int]): (list[int]) List of person IDs for filtering the DataFrame.
        years (list[int]): (list[int]) List of years for processing the DataFrame.
        folder (str): (str) Folder name under FPATH.DATA where the output file will be saved.
        min_occurrence_time (int): (int) Minimum number of occurrences for a course to have its own column.
        force_copy (bool): (bool) Whether to force copying of file from the opposite drive.
    """

    # Copy if forced to
    if force_copy:
        copy_file_or_dir(FPATH.DUMP_DIR / f"{filename}.parquet")
    else:
        check_and_copy_file_or_dir(FPATH.DUMP_DIR / f"{filename}.parquet")

    # Load the full unfiltered DataFrame.
    df_udfk = pl.read_parquet(FPATH.DUMP_DIR / f"{filename}.parquet")

    # Identify frequent courses based on min_occurrence_time.
    course_counts = df_udfk.group_by("course").agg(pl.count("course").alias("count"))
    frequent_courses = course_counts.filter(pl.col("count") >= min_occurrence_time)[
        "course"
    ].to_list()

    # Filter for specified person IDs.
    relevant_events = df_udfk.filter(pl.col("person_id").is_in(person_ids))

    # Free up memory
    del df_udfk

    udfk_dataframes: list[pl.DataFrame] = []
    for year in years:
        # Filter events that occurred before the current year.
        previous_events = relevant_events.filter(pl.col("year").cast(pl.Int64) < year)

        dataframe_person_id_year = (
            pl.DataFrame({"person_id": previous_events["person_id"].unique().to_list()})
            .cast({"person_id": int})
            .with_columns(pl.lit(year).alias("year"))
        )

        # Compute mean grade with separate columns for frequent courses and a pooled column for infrequent courses.
        result_frequent_df, result_infrequent_df = calculate_mean_grade_pool_infrequent(
            previous_events,
            "person_id",
            "course",
            "grade",
            alias_prefix="ps_mean_",
            frequent_courses=frequent_courses,
        )

        result = dataframe_person_id_year.join(
            result_frequent_df, how="left", on="person_id"
        ).join(result_infrequent_df, how="left", on="person_id")

        udfk_dataframes.append(result)

    all_udfk = pl.concat(udfk_dataframes, how="diagonal_relaxed")
    output_file = FPATH.DATA / folder / "udfk.parquet"
    all_udfk.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_univid(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    min_occurrence_time: int,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the UNIVID dataset filtered by person IDs and years.

    The function loads the full unfiltered DataFrame, identifies courses occurring at least
    `min_occurrence_time`, and computes weighted mean grades. Frequent courses get their own weighted mean columns,
    while the remaining courses are averaged into a single overall weighted mean column.

    Args:
        filename (str): (str) Base filename (e.g., "karakter_univid_fag_DUMP") without extension.
        person_ids (list[int]): (list[int]) List of person IDs for filtering the DataFrame.
        years (list[int]): (list[int]) List of years for processing the DataFrame.
        folder (str): (str) Folder name under FPATH.DATA where the output file will be saved.
        min_occurrence_time (int): (int) Minimum number of occurrences for a course to have its own column.
        force_copy (bool): (bool) Whether to force copying of the file from the opposite drive.
    """
    # Copy if forced to
    if force_copy:
        copy_file_or_dir(FPATH.DUMP_DIR / f"{filename}.parquet")
    else:
        check_and_copy_file_or_dir(FPATH.DUMP_DIR / f"{filename}.parquet")

    # Load the full unfiltered DataFrame.
    df_univid = pl.read_parquet(
        FPATH.swap_drives(FPATH.DUMP_DIR / f"{filename}.parquet")
    )

    # Identify frequent courses based on min_occurrence_time.
    course_counts = df_univid.group_by("kurstxt").agg(
        pl.count("kurstxt").alias("count")
    )
    frequent_courses = course_counts.filter(pl.col("count") >= min_occurrence_time)[
        "kurstxt"
    ].to_list()

    # Filter for specified person IDs.
    relevant_events = df_univid.filter(pl.col("person_id").is_in(person_ids))

    # Free up memory
    del df_univid

    univid_dataframes = []
    for year in years:
        # Filter events that occurred before the current year.
        previous_events = relevant_events.filter(
            pl.col("bedommelsesdato").dt.year() < year
        )

        dataframe_person_id_year = (
            pl.DataFrame({"person_id": previous_events["person_id"].unique().to_list()})
            .cast({"person_id": int})
            .with_columns(pl.lit(year).alias("year"))
        )

        # Compute weighted mean grade with pooling for infrequent courses.
        result_frequent_df, result_infrequent_df = (
            calculate_weighted_mean_grade_pool_infrequent(
                previous_events,
                "person_id",
                "kurstxt",
                "grade",
                "ects",
                alias_prefix="he_mean_",
                frequent_courses=frequent_courses,
            )
        )
        modes = calculate_mode_by_personid(
            previous_events, ["instnr", "udd"], "he_mode_"
        )

        result = (
            dataframe_person_id_year.join(
                result_frequent_df, how="left", on="person_id"
            )
            .join(result_infrequent_df, how="left", on="person_id")
            .join(modes, how="left", on="person_id")
        )
        univid_dataframes.append(result)

    all_univid = pl.concat(univid_dataframes, how="diagonal_relaxed")
    output_file = FPATH.DATA / folder / "univid.parquet"
    all_univid.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_udd(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    field_truncation_level: int,
    disced_truncation_level: int,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the UDD dataset filtered by person IDs and years.

    Args:
        filename (str): Base filename (e.g., "karakter_udd_DUMP_AUGMENTED") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        field_truncation_level (int): Number of characters to retain for the 'field' column.
        disced_truncation_level (int): Number of characters to retain for the 'disced' column.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """
    df_udd = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    )

    # Load the mapping file
    mapping_path = FPATH.swap_drives(FPATH.DATA / "mappings" / "13_to_7_mapping.txt")
    with open(mapping_path, "r") as f:
        map_13_to_7 = json.load(f)

    df_udd = df_udd.with_columns(
        pl.when(df_udd["skala"] == 13)
        .then(
            df_udd["karakter_udd"].map_elements(
                lambda x: map_13_to_7.get(str(x)), return_dtype=pl.Utf8
            )
        )
        .otherwise(df_udd["karakter_udd"])
        .alias("gpa")
        .cast(pl.Int64)
        .mul(1 / 10)
    )

    df_udd = df_udd.with_columns(
        pl.col("disced")
        .cast(pl.Utf8)
        .str.slice(0, disced_truncation_level)
        .alias("disced"),
        pl.col("field")
        .cast(pl.Utf8)
        .str.slice(0, field_truncation_level)
        .alias("field"),
    ).select(
        ["person_id", "karakter_udd_vtil", "audd", "instnr", "disced", "field", "gpa"]
    )

    udd_dataframes = []
    for year in years:
        previous_events = df_udd.filter(pl.col("karakter_udd_vtil").dt.year() < year)
        result = (
            previous_events.sort("karakter_udd_vtil")
            .group_by("person_id")
            .tail(1)
            .drop("karakter_udd_vtil")
        )
        result = result.with_columns(pl.lit(year).alias("year"))
        udd_dataframes.append(result)
    all_udd = pl.concat(udd_dataframes, how="vertical")

    output_file = FPATH.DATA / folder / "udd.parquet"
    all_udd.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_kraf(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    ger7_truncation_level: int,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save various KRAF datasets (revoked license, suspended sentences,
    convictions/no convictions, and fines) filtered by person IDs and years.

    Args:
        filename (str): Base filename (e.g., "kraf_DUMP_AUGMENTED") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output files will be saved.
        ger7_truncation_level (int): Number of characters to retain for the 'ger7' column.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """

    # Load kraf data and truncate 'ger7'
    df_kraf = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    ).with_columns(
        pl.col("ger7").cast(pl.Utf8).str.slice(0, ger7_truncation_level).alias("ger7")
    )

    # ------------------
    # Drivers license
    df_license = df_kraf[["person_id", "afgoedto", "revoke_license_days"]].filter(
        pl.col("revoke_license_days").is_not_null()
    )
    revoked_license_frames = []
    for cutoff_year in years:
        target_date = datetime(cutoff_year, 1, 1)
        previous_events = df_license.filter(pl.col("afgoedto").dt.year() < cutoff_year)
        results = filter_and_flag_events(
            previous_events,
            target_date,
            "afgoedto",
            "revoke_license_days",
            "currently_revoked_license",
            "has_had_revoked_licenses",
        )
        results = results.with_columns(pl.lit(cutoff_year).alias("year"))
        revoked_license_frames.append(results)
    df_revoked_license = pl.concat(revoked_license_frames)
    output_file = FPATH.DATA / folder / "revoked_license.parquet"
    df_revoked_license.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)

    # ------------------
    # Suspended sentences
    df_susp_jail = df_kraf[
        ["person_id", "afgoedto", "betbkod", "cond_jail_days"]
    ].filter(pl.col("cond_jail_days").is_not_null() | pl.col("betbkod").is_not_null())
    susp_jail_frames = []
    for cutoff_year in years:
        target_date = datetime(cutoff_year, 1, 1)
        previous_events = df_susp_jail.filter(
            pl.col("afgoedto").dt.year() < cutoff_year
        )
        results = filter_and_flag_events_with_binary(
            previous_events,
            target_date,
            "afgoedto",
            "cond_jail_days",
            "betbkod",
            "curr_susp_sent",
            "has_had_susp_sent",
            "perpetual_susp_sent",
        )
        results = results.with_columns(pl.lit(cutoff_year).alias("year"))
        susp_jail_frames.append(results)
    df_susp_jail_results = pl.concat(susp_jail_frames)
    output_file = FPATH.DATA / folder / "suspended_jail.parquet"
    df_susp_jail_results.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)

    # ------------------
    # Crimes: Convictions and No Convictions
    df_crimes = df_kraf[["person_id", "afgoedto", "ger7", "afgtyp3"]]
    no_convictions_frames = []
    convictions_frames = []
    free_cols = [0, 514, 518, 519]
    for year in years:
        previous_events = df_crimes.filter(pl.col("afgoedto").dt.year() < year)
        # No convictions (e.g., 'Frifundet')
        no_convictions = previous_events.filter(pl.col("afgtyp3").is_in(free_cols))
        result_no_convictions = group_values_multiple(
            no_convictions, "person_id", ["ger7"]
        )
        result_no_convictions = result_no_convictions.with_columns(
            pl.lit(year).alias("year")
        ).rename({"ger7": "ger7_no_conviction"})
        no_convictions_frames.append(result_no_convictions)
        # Convictions
        convictions = previous_events.filter(~pl.col("afgtyp3").is_in(free_cols))
        result_convictions = group_values_multiple(convictions, "person_id", ["ger7"])
        result_convictions = result_convictions.with_columns(
            pl.lit(year).alias("year")
        ).rename({"ger7": "ger7_conviction"})
        convictions_frames.append(result_convictions)
    df_no_convictions = pl.concat(no_convictions_frames)
    df_convictions = pl.concat(convictions_frames)
    output_file = FPATH.DATA / folder / "convictions.parquet"
    df_convictions.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)
    output_file = FPATH.DATA / folder / "no_convictions.parquet"
    df_no_convictions.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)

    # ------------------
    # Fines
    df_fines = df_kraf[["person_id", "afgoedto", "boedeblb", "dagboant", "dagbobel"]]
    fines_frames = []
    for cutoff_year in years:
        filtered_df = df_fines.filter(pl.col("afgoedto").dt.year() < cutoff_year)
        result_df = (
            filtered_df.group_by("person_id").agg(
                [
                    pl.col("boedeblb").sum().alias("boedeblb_sum"),
                    pl.col("dagboant").mean().alias("dagboant_mean"),
                    pl.col("dagbobel").mean().alias("dagbobel_mean"),
                ]
            )
        ).with_columns(pl.lit(cutoff_year).alias("year"))
        fines_frames.append(result_df)
    df_fines_final = pl.concat(fines_frames)
    output_file = FPATH.DATA / folder / "fines.parquet"
    df_fines_final.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_krin(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the KRIN dataset filtered by person IDs and years.

    Args:
        filename (str): Base filename (e.g., "krin_DUMP") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """

    df_krin = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    )

    lst_jailings = []
    for cutoff_year in years:
        target_date = datetime(cutoff_year, 1, 1)
        previous_events = df_krin.filter(pl.col("ind_fgsldto").dt.year() < cutoff_year)
        results = filter_and_flag_events_with_end_date(
            previous_events,
            target_date,
            "ind_fgsldto",
            "ind_loesldto",
            "in_jail",
            "has_been_in_jail",
        )
        results = results.with_columns(pl.lit(cutoff_year).alias("year"))
        lst_jailings.append(results)

    df_jailed = pl.concat(lst_jailings)
    output_file = FPATH.DATA / folder / "jailed.parquet"
    df_jailed.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_krko(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    ger7_truncation_level: int,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the KRKO dataset related to confederated cases filtered by person IDs and years.

    Args:
        filename (str): Base filename (e.g., "krko_DUMP") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output files will be saved.
        ger7_truncation_level (int): Number of characters to retain for the 'ger7' column.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """
    df_krko = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    ).with_columns(
        pl.col("ger7").cast(pl.Utf8).str.slice(0, ger7_truncation_level).alias("ger7")
    )

    df_confederated_cases = df_krko[["person_id", "afg_afgoedto", "ger7", "afgtypko"]]

    lst_confederated_no_convictions = []
    lst_confederated_convictions = []
    for year in years:
        previous_events = df_confederated_cases.filter(
            pl.col("afg_afgoedto").dt.year() < year
        )
        free_cols = [0, 11, 514, 518, 519]
        confederated_no_convictions = previous_events.filter(
            pl.col("afgtypko").is_in(free_cols)
        )
        result_confederated_no_convictions = group_values_multiple(
            confederated_no_convictions, "person_id", ["ger7"]
        )
        result_confederated_no_convictions = (
            result_confederated_no_convictions.with_columns(
                pl.lit(year).alias("year")
            ).rename({"ger7": "ger7_conf_no_convs"})
        )
        lst_confederated_no_convictions.append(result_confederated_no_convictions)

        confederated_convictions = previous_events.filter(
            ~pl.col("afgtypko").is_in(free_cols)
        )
        result_confederated_convictions = group_values_multiple(
            confederated_convictions, "person_id", ["ger7"]
        ).rename({"ger7": "ger7_conf_convs"})
        result_confederated_convictions = result_confederated_convictions.with_columns(
            pl.lit(year).alias("year")
        )
        lst_confederated_convictions.append(result_confederated_convictions)

    df_confederated_no_convictions = pl.concat(lst_confederated_no_convictions)
    df_confederated_convictions = pl.concat(lst_confederated_convictions)

    output_file = FPATH.DATA / folder / "confederated_convictions.parquet"
    df_confederated_convictions.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)

    output_file = FPATH.DATA / folder / "confederated_no_convictions.parquet"
    df_confederated_no_convictions.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_krsi(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    ger7_truncation_level: int,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the KRSI dataset for charges filtered by person IDs and years.

    Args:
        filename (str): Base filename (e.g., "krsi_DUMP_AUGMENTED") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        ger7_truncation_level (int): Number of characters to retain for the 'ger7' column.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """
    df_krsi = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    ).with_columns(
        pl.col("ger7").cast(pl.Utf8).str.slice(0, ger7_truncation_level).alias("ger7")
    )

    lst_charges = []
    for year in years:
        previous_events = df_krsi.filter(pl.col("sigtdto").dt.year() < year)
        results = group_values_multiple(previous_events, "person_id", ["ger7"])
        results = results.with_columns(pl.lit(year).alias("year")).rename(
            {"ger7": "ger7_charges"}
        )
        lst_charges.append(results)

    df_charges = pl.concat(lst_charges)

    output_file = FPATH.DATA / folder / "charges.parquet"
    df_charges.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_krof(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    ger7_truncation_level: int,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the KROF dataset for victims filtered by person IDs and years.

    Args:
        filename (str): Base filename (e.g., "krof_DUMP_AUGMENTED") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        ger7_truncation_level (int): Number of characters to retain for the 'ger7' column.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """
    df_krof = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    ).with_columns(
        pl.col("ger7").cast(pl.Utf8).str.slice(0, ger7_truncation_level).alias("ger7")
    )

    lst_victims = []
    for year in years:
        previous_events = df_krof.filter(pl.col("ofr_gerfradt").dt.year() < year)
        results = group_values_multiple(previous_events, "person_id", ["ger7"])
        results = results.with_columns(pl.lit(year).alias("year")).rename(
            {"ger7": "ger7_victims"}
        )
        lst_victims.append(results)

    df_victims = pl.concat(lst_victims)

    output_file = FPATH.DATA / folder / "victims.parquet"
    df_victims.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_krms(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    ger7_truncation_level: int,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the KRMS dataset for minor charges filtered by person IDs and years.

    Args:
        filename (str): Base filename (e.g., "krms_DUMP_AUGMENTED") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        ger7_truncation_level (int): Number of characters to retain for the 'ger7' column.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """
    df_krms = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    ).with_columns(
        pl.col("ger7").cast(pl.Utf8).str.slice(0, ger7_truncation_level).alias("ger7")
    )

    lst_minor_charges = []
    for year in years:
        previous_events = df_krms.filter(pl.col("sigtdto").dt.year() < year)
        results = group_values_multiple(previous_events, "person_id", ["ger7"])
        results = results.with_columns(pl.lit(year).alias("year")).rename(
            {"ger7": "ger7_minor_charge"}
        )
        lst_minor_charges.append(results)

    df_minor_charges = pl.concat(lst_minor_charges)

    output_file = FPATH.DATA / folder / "minor_charges.parquet"
    df_minor_charges.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_dnt(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the LMDB dataset filtered by person IDs and years.

    Args:
        filename (str): Base filename (e.g., "lmdb_DUMP") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """
    df_dnt = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    )

    dnt_dataframes = []
    for year in years:
        filtered = df_dnt.filter(pl.col("event_date").dt.year() < year)

        if filtered.shape[0] == 0:
            continue

        pivoted = (
            filtered.select(
                ["person_id", "fag", "point_samlet_normalized_by_year_course"]
            )
            .pivot(
                values="point_samlet_normalized_by_year_course",
                index="person_id",
                on="fag",
                aggregate_function="first",
            )
            .with_columns(pl.lit(year).alias("year"))
        )

        dnt_dataframes.append(pivoted)

    all_dnt = pl.concat(dnt_dataframes, how="diagonal_relaxed")

    output_file = FPATH.DATA / folder / "dnt.parquet"
    all_dnt.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_wellbeing(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the wellbeing dataset filtered by person IDs and years.
    Keeps only the latest observation before each year.

    Args:
        filename (str): Base filename (e.g., "lmdb_DUMP") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """
    df_wellbeing = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    )

    wellbeing_dataframes = []
    for year in years:
        filtered = df_wellbeing.filter(pl.col("event_date").dt.year() < year)
        latest_per_person = (
            filtered.sort("event_date", descending=True)
            .drop("event_date")
            .unique(subset=["person_id"], keep="first")
            .with_columns(pl.lit(year).alias("year"))
        )

        wellbeing_dataframes.append(latest_per_person)

    all_wellbeing = pl.concat(wellbeing_dataframes)

    output_file = FPATH.DATA / folder / "wellbeing.parquet"
    all_wellbeing.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_hoejste_fuldfoert_udd(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    truncate_disced_level: int,
    truncate_disced_field: int,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the highest completed education dataset, filtered by person IDs and years.
    For each year, selects the latest row per person based on hf_vfra < year. Optionally truncates disced codes.

    Args:
        filename (str): Base filename (e.g., "lmdb_DUMP") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        truncate_disced_level (int): Number of characters to keep from audd_disced.
        truncate_disced_field (int): Number of characters to keep from audd_field.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """
    df_udd = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    )

    udd_dataframes = []
    for year in years:
        filtered = df_udd.filter(pl.col("hf_vfra").dt.year() < year)

        latest_per_person = (
            filtered.sort("hf_vfra", descending=True)
            .unique(subset=["person_id"], keep="first")
            .with_columns(
                [
                    pl.col("audd_disced").str.slice(0, truncate_disced_level),
                    pl.col("audd_field").str.slice(0, truncate_disced_field),
                    pl.lit(year).alias("year"),
                ]
            )
            .select(["person_id", "hfaudd", "audd_disced", "audd_field", "year"])
        )

        udd_dataframes.append(latest_per_person)

    all_udd = pl.concat(udd_dataframes)

    output_file = FPATH.DATA / folder / "hoejste_fuldfoert_udd.parquet"
    all_udd.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_elev3(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    truncate_disced: int,
    truncate_field: int,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the elev3 dataset, filtered by person IDs and years.
    For each year, finds rows where Jan 1 of the year is between elev3_vfra and elev3_vtil.
    Selects the last row per person, includes an indicator, and truncates udd_disced and udd_field.

    Args:
        filename (str): Base filename (e.g., "lmdb_DUMP") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        truncate_disced (int): Number of characters to keep from udd_disced.
        truncate_field (int): Number of characters to keep from udd_field.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """
    df_elev3 = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    )

    elev3_dataframes = []
    for year in years:
        dec31 = pl.datetime(year - 1, 12, 31)

        filtered = df_elev3.filter(
            (pl.col("elev3_vfra") <= dec31) & (pl.col("elev3_vtil") >= dec31)
        )

        latest_per_person = (
            filtered.sort("elev3_vfra", descending=True)
            .unique(subset=["person_id"], keep="first")
            .with_columns(
                [
                    pl.lit(1).alias("currently_enrolled"),
                    pl.col("udd_disced").str.slice(0, truncate_disced),
                    pl.col("udd_field").str.slice(0, truncate_field),
                    pl.lit(year).alias("year"),
                ]
            )
            .select(
                [
                    "person_id",
                    "currently_enrolled",
                    "udd",
                    "udel",
                    "udd_disced",
                    "udd_field",
                    "year",
                ]
            )
        )

        elev3_dataframes.append(latest_per_person)

    all_elev3 = pl.concat(elev3_dataframes)

    output_file = FPATH.DATA / folder / "elev3.parquet"
    all_elev3.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_absences(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the fravaer_stil dataset, filtered by person IDs and years.
    For each year,find mean fraction of absences across absence types.

    Args:
        filename (str): Base filename (e.g., "lmdb_DUMP") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """
    df_absences = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    )

    absences_dataframes = []
    absence_cols = ["dageialtfra", "dagelovfra", "dagesyg", "dageulovfra"]

    df_absences = df_absences.with_columns(
        [(pl.col(col) / pl.col("dageaktiv")).alias(col) for col in absence_cols]
    )
    for year in years:
        dec31_this_year = pl.datetime(year - 1, 12, 31)
        dec31_previous_year = pl.datetime(year - 1, 12, 31)

        filtered = df_absences.filter(
            (pl.col("event_date") <= dec31_this_year)
            & (pl.col("event_date") >= dec31_previous_year)
        )

        mean_per_person = (
            filtered.group_by("person_id")
            .mean()
            .select(["person_id", "dageaktiv"] + absence_cols)
        ).with_columns(pl.lit(year).alias("year"))

        absences_dataframes.append(mean_per_person)

    all_absences = pl.concat(absences_dataframes)

    output_file = FPATH.DATA / folder / "absences.parquet"
    all_absences.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_buaf(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the BUAF dataset, filtered by person IDs and years.
    For each year, checks if the person was in out-of-home placement on Dec 31 of the previous year
    and whether they were ever placed before that point.

    Args:
        filename (str): Base filename (e.g., "buaf_DUMP") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """
    df_buaf = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    )

    buaf_dataframes = []
    for year in years:
        dec31 = pl.datetime(year - 1, 12, 31)

        # Currently placed on Dec 31
        current = (
            df_buaf.filter(
                (pl.col("startdato") <= dec31)
                & ((pl.col("slutdato").is_null()) | (pl.col("slutdato") >= dec31))
            )
            .select(["person_id"])
            .unique()
            .with_columns(
                [
                    pl.lit(1).alias("currently_out_of_home"),
                ]
            )
            .select(["person_id", "currently_out_of_home"])
        )

        # Ever placed before Dec 31
        ever = (
            df_buaf.filter(pl.col("startdato") <= dec31)
            .select(["person_id"])
            .unique()
            .with_columns(pl.lit(1).alias("has_been_out_of_home"))
            .select(["person_id", "has_been_out_of_home"])
        )

        previous_events = df_buaf.filter(pl.col("startdato").dt.year() < year)
        result = group_values_multiple(
            previous_events, "person_id", ["samtykke", "ansted_klas", "haendelse"]
        )
        result = (
            result.with_columns(pl.lit(year).alias("year"))
            .join(current, how="left", on="person_id")
            .join(ever, how="left", on="person_id")
        )

        result = result.fill_null(0)

        buaf_dataframes.append(result)

    all_buaf = pl.concat(buaf_dataframes)

    output_file = FPATH.DATA / folder / "buaf.parquet"
    all_buaf.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_bufo(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the BUFO dataset, filtered by person IDs and years.
    For each year, checks if the person was had preventive measures on Dec 31 of the previous year
    and whether they have ever had preventive measures.

    Args:
        filename (str): Base filename (e.g., "bufo_DUMP") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """
    df_bufo = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    )

    bufo_dataframes = []
    for year in years:
        dec31 = pl.datetime(year - 1, 12, 31)

        current = (
            df_bufo.filter(
                (pl.col("sag_vfra") <= dec31)
                & ((pl.col("sag_vtil").is_null()) | (pl.col("sag_vtil") >= dec31))
            )
            .select(["person_id"])
            .unique()
            .with_columns(pl.lit(1).alias("currently_preventive_measures"))
        )

        ever = (
            df_bufo.filter(pl.col("sag_vfra") <= dec31)
            .select(["person_id"])
            .unique()
            .with_columns(pl.lit(1).alias("has_had_preventive_measures"))
        )

        previous_events = df_bufo.filter(pl.col("sag_vfra").dt.year() < year)
        result = group_values_multiple(previous_events, "person_id", ["pgf"])

        result = (
            result.with_columns(pl.lit(year).alias("year"))
            .join(current, how="left", on="person_id")
            .join(ever, how="left", on="person_id")
            .fill_null(0)
        )

        bufo_dataframes.append(result)

    all_bufo = pl.concat(bufo_dataframes)
    output_file = FPATH.DATA / folder / "bufo.parquet"
    all_bufo.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_bef(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    force_copy: bool = False,
) -> None:
    """
    Load, filter, and save the BEF dataset, keeping the latest record before each year for each person.

    Args:
        filename (str): Base filename (e.g., "bef_DUMP") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """
    df_bef = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    )

    df_lifelines = (
        collect_filtered_parquet(
            FPATH.DUMP_DIR / "lifelines_births_deaths.parquet",
            person_ids,
            force_copy=force_copy,
        )
        .filter(pl.col("event") == "Birth")
        .rename({"event_date": "birthday"})
        .select(["person_id", "birthday"])
    )

    df_bef = df_bef.join(df_lifelines, how="left", on="person_id")

    keep_cols = [
        "person_id",
        "age",
        # "referencetid",
        "antboernf",
        "antpersf",
        "antefam",
        "civst",
        "fm_mark",
        "statsb",
        "kom",
    ]

    bef_dataframes = []
    for year in years:
        ts = pl.datetime(year, 1, 1)

        latest = (
            df_bef.filter(pl.col("referencetid") < ts)
            .with_columns(
                ((ts - pl.col("birthday")).dt.total_days() // 365)
                .cast(pl.Int32)
                .alias("age")
            )
            .sort("referencetid", descending=True)
            .group_by("person_id", maintain_order=True)
            .agg([pl.first(col) for col in keep_cols if col != "person_id"])
            .with_columns(pl.lit(year).alias("year"))
        )

        bef_dataframes.append(latest)

    all_bef = pl.concat(bef_dataframes)

    all_bef = all_bef.with_columns(
        (pl.col("antefam") > 1).cast(int).alias("multi_fam_household")
    )
    output_file = FPATH.DATA / folder / "bef.parquet"
    all_bef.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_smdb_ibib(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the SMDB_IBIB dataset, filtered by person IDs and years.
    For each year, checks if the person was in drug abuse treatment on Dec 31 of the previous year
    and whether they had ever been treated before that point.

    Args:
        filename (str): Base filename (e.g., "smdb_ibib_DUMP") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """
    df_ibib = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    )

    ibib_dataframes = []
    for year in years:
        dec31 = pl.datetime(year - 1, 12, 31)

        current = (
            df_ibib.filter(
                (pl.col("ydelsestartdato") <= dec31)
                & (
                    (pl.col("ydelseslutdato").is_null())
                    | (pl.col("ydelseslutdato") >= dec31)
                )
            )
            .select(["person_id"])
            .unique()
            .with_columns(pl.lit(1).alias("currently_in_treatment"))
        )

        ever = (
            df_ibib.filter(pl.col("ydelsestartdato") <= dec31)
            .select(["person_id"])
            .unique()
            .with_columns(pl.lit(1).alias("has_been_in_treatment"))
        )

        result = (
            pl.DataFrame({"person_id": person_ids})
            .join(current, how="left", on="person_id")
            .join(ever, how="left", on="person_id")
            .with_columns(pl.lit(year).alias("year"))
            .fill_null(0)
        )

        ibib_dataframes.append(result)

    all_ibib = pl.concat(ibib_dataframes)
    output_file = FPATH.DATA / folder / "smdb_ibib.parquet"
    all_ibib.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_emigration(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    force_copy: bool = False,
) -> None:
    """
    Load, filter, and save the lifelines immigration/emigration dataset, indicating whether
    each person has ever emigrated before the given year.

    Args:
        filename (str): Base filename (e.g., "lifelines_immigration_emmigration") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """
    df_life = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    )

    emm_dataframes = []
    df_emm = df_life.filter(pl.col("event") == "Udvandret")

    for year in years:
        ts = pl.datetime(year, 1, 1)

        ever = (
            df_emm.filter(pl.col("event_date") < ts)
            .select(["person_id"])
            .unique()
            .with_columns(pl.lit(1).alias("has_ever_been_emmigrated"))
        )

        result = (
            pl.DataFrame({"person_id": person_ids})
            .join(ever, how="left", on="person_id")
            .with_columns(pl.lit(year).alias("year"))
            .fill_null(0)
        )

        emm_dataframes.append(result)

    all_emm = pl.concat(emm_dataframes)
    output_file = FPATH.DATA / folder / "emigration.parquet"
    all_emm.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_befbop(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the BEFBOP dataset with moving indicators.
    Adds whether a person moved within the previous year and counts total moves before each year.

    Args:
        filename (str): Base filename (e.g., "befbop_DUMP") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """
    df_bop = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    )

    bop_dataframes = []
    for year in years:
        jan1 = pl.datetime(year, 1, 1)
        jan1_prev = pl.datetime(year - 1, 1, 1)

        moved_last_year = (
            df_bop.filter(
                (pl.col("bop_vfra") >= jan1_prev) & (pl.col("bop_vfra") < jan1)
            )
            .select(["person_id"])
            .unique()
            .with_columns(pl.lit(1).alias("moved_within_last_year"))
        )

        total_moves = (
            df_bop.filter(pl.col("bop_vfra") < jan1)
            .group_by("person_id")
            .agg(pl.count().alias("total_previous_moves"))
        )

        result = (
            pl.DataFrame({"person_id": person_ids})
            .join(moved_last_year, how="left", on="person_id")
            .join(total_moves, how="left", on="person_id")
            .with_columns(pl.lit(year).alias("year"))
            .fill_null(0)
        )

        bop_dataframes.append(result)

    all_bop = pl.concat(bop_dataframes)
    output_file = FPATH.DATA / folder / "befbop.parquet"
    all_bop.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)


def preprocess_family_information(
    filename: str,
    person_ids: list[int],
    years: list[int],
    folder: str,
    force_copy: bool = False,
) -> None:
    """
    Load, transform, and save the family events dataset.
    Computes family structure and death indicators for each person-year,
    using only unique person_ids with events before the yearly cutoff.

    Args:
        filename (str): Base filename (e.g., "family_events") without extension.
        person_ids (list[int]): List of person IDs used for filtering the DataFrame.
        years (list[int]): List of years used for processing the DataFrame.
        folder (str): Folder name under FPATH.DATA where the output file will be saved.
        force_copy (bool): Whether to force copy of file from opposite drive. Default False.
    """
    df_family = collect_filtered_parquet(
        FPATH.DUMP_DIR / f"{filename}.parquet", person_ids, force_copy=force_copy
    )

    family_dataframes = []
    for year in years:
        ts = pl.datetime(year, 1, 1)
        df = df_family.filter(pl.col("event_date") < ts)
        if df.shape[0] == 0:
            continue

        year_person_ids = df.select("person_id").unique()

        # Siblings
        siblings = df.filter(
            pl.col("relation_details").is_in(
                ["Full sibling", "Maternal half sibling", "Paternal half sibling"]
            )
        )
        siblings_alive = siblings.filter(pl.col("event") == "Birth")
        siblings_dead = siblings.filter(pl.col("event") == "Death")

        sibling_counts = siblings_alive.group_by("person_id").agg(
            pl.len().alias("num_siblings")
        )
        dead_sibling_counts = siblings_dead.group_by("person_id").agg(
            pl.len().alias("num_dead_siblings")
        )

        sibling_features = (
            year_person_ids.join(sibling_counts, on="person_id", how="left")
            .join(dead_sibling_counts, on="person_id", how="left")
            .with_columns(
                [
                    (pl.col("num_siblings").fill_null(0)),
                    (pl.col("num_dead_siblings").fill_null(0)),
                    (pl.col("num_siblings") - pl.col("num_dead_siblings")).alias(
                        "num_alive_siblings"
                    ),
                ]
            )
        )

        # Children
        children = df.filter(pl.col("relation_details") == "Child")
        children_alive = children.filter(pl.col("event") == "Birth")
        children_dead = children.filter(pl.col("event") == "Death")

        child_counts = children_alive.group_by("person_id").agg(
            pl.len().alias("num_children")
        )
        dead_child_counts = children_dead.group_by("person_id").agg(
            pl.len().alias("num_dead_children")
        )

        child_features = (
            year_person_ids.join(child_counts, on="person_id", how="left")
            .join(dead_child_counts, on="person_id", how="left")
            .with_columns(
                [
                    (pl.col("num_children").fill_null(0)),
                    (pl.col("num_dead_children").fill_null(0)),
                    (pl.col("num_children") - pl.col("num_dead_children")).alias(
                        "num_alive_children"
                    ),
                ]
            )
        )

        # Parents
        parent_deaths = df.filter(
            (pl.col("relation_details").is_in(["Mother", "Father"]))
            & (pl.col("event") == "Death")
        )
        parent_flags = parent_deaths.pivot(
            values="event",
            index="person_id",
            columns="relation_details",
            aggregate_function="len",
        ).fill_null(0)

        select_cols = ["person_id"]
        if "Mother" in parent_flags.columns:
            parent_flags = parent_flags.with_columns(
                (pl.col("Mother") > 0).cast(pl.Int8).alias("is_mother_dead")
            )
            select_cols.append("is_mother_dead")
        if "Father" in parent_flags.columns:

            parent_flags = parent_flags.with_columns(
                (pl.col("Father") > 0).cast(pl.Int8).alias("is_father_dead")
            )
            select_cols.append("is_father_dead")

        parent_flags = parent_flags.select(select_cols)

        parent_features = year_person_ids.join(
            parent_flags, on="person_id", how="left"
        ).fill_null(0)

        # Final join
        result = (
            year_person_ids.join(sibling_features, on="person_id", how="left")
            .join(child_features, on="person_id", how="left")
            .join(parent_features, on="person_id", how="left")
            .with_columns(pl.lit(year).alias("year"))
            .fill_null(0)
        )

        family_dataframes.append(result)

    all_family = pl.concat(family_dataframes, how="diagonal_relaxed")
    output_file = FPATH.DATA / folder / "family_information.parquet"
    all_family.write_parquet(output_file)
    FPATH.alternative_copy_to_opposite_drive(output_file)
