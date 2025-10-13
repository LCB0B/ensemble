# %%
import matplotlib.pyplot as plt

plt.plot([1,2,3])

# %%
import os
from src.paths import FPATH
os.makedirs(FPATH.DATA / "destiny" / "outcomes", exist_ok=True)
os.makedirs(FPATH.NETWORK_DATA / "destiny" / "cohort", exist_ok=True)

os.environ["POLARS_MAX_THREADS"] = "8"
os.environ["RAYON_NUM_THREADS"] = "8"
import polars as pl

# %%
# Load cohort files if they exist, otherwise they'll be created below
cohort_path = FPATH.NETWORK_DATA / "destiny" / "cohort"

if (cohort_path / "fullpop.parquet").exists():
    print("Loading existing cohort files...")
    fullpop = pl.read_parquet(cohort_path / "fullpop.parquet")
    periods = pl.read_parquet(cohort_path / "periods.parquet")
    train = pl.read_parquet(cohort_path / "train.parquet")
    val = pl.read_parquet(cohort_path / "val.parquet")
    test = pl.read_parquet(cohort_path / "test.parquet")
    print(f"Loaded cohort with {len(fullpop):_} people")
else:
    print("Cohort files not found. Creating them now...")
    # The cohort creation cells below (lines ~88-149) will create these files
    fullpop, periods, train, val, test = None, None, None, None, None


def get_positives(df, column, codes, fullpop, bday_count=10):
    # Define positives
    positives = df.filter(
        pl.any_horizontal(pl.col(column).str.starts_with(code) for code in codes)
    ).sort("date_col").group_by("person_id").first().join(fullpop, on="person_id")
    print(f"{'Number positives:':<20} {len(positives):_}")

    # Filter on birthday count
    valids = positives.select(pl.col("birthday").dt.year())["birthday"].value_counts().filter(pl.col("count") >= bday_count)
    positives = positives.filter(pl.col("birthday").dt.year().is_in(valids["birthday"]))
    print(f"{'Valid positives:':<20} {len(positives):_}")

    return positives

def get_cohort(fullpop, positives, periods, year_reduce=5):
    cohort = fullpop.join(positives.select("person_id", outcome="date_col"), on="person_id", how="left")
    print(f"{'Starting cohort:':<20} {len(cohort):_}")
    
    # Apply approximate cohort
    bmin, bmax = positives["birthday"].dt.year().min(), positives["birthday"].dt.year().max()
    cohort = cohort.filter((pl.col("birthday").dt.year() >= bmin) & (pl.col("birthday").dt.year() <= bmax))

    # Draw random censoring/prediction time to no-outcome people based on birthyear (with year_reduce)
    cohort = cohort.with_columns(year=pl.col("birthday").dt.year() // year_reduce).group_by("year").map_groups(
        lambda group:
        group.with_columns(censor=pl.col("outcome").fill_null(
            pl.col("outcome").drop_nulls().sample(pl.len(), with_replacement=True)
        ))                                                                 
    )

    # Filter only prediction times inside a valid period
    cohort = cohort.join(periods, on="person_id").filter(
        (pl.col("censor") >= pl.col("event_start_date")) & (pl.col("censor") <= pl.col("event_final_date"))
    ).drop("event_start_date", "event_final_date")
    
    cohort = cohort.with_columns(
        predict=pl.concat_list([
            pl.when(pl.col("censor").dt.offset_by(offset) > pl.col("birthday"))
            .then(pl.col("censor").dt.offset_by(offset)).otherwise(None)
            for offset in ["-10y", "-5y", "-3y", "-1y"]
        ]).list.drop_nulls()
    ).filter(pl.col("predict").list.len() > 0)
    print(f"{'Final cohort:':<20} {len(cohort):_}")

    c_string = f"{'Train/val/test:':<20} " + '  /  '.join(
        f"{len(subset):_} ({len(subset.filter(pl.col('outcome').is_not_null())):_})" 
        for subset in [cohort.join(s, on="person_id") for s in [train, val, test]]
    )
    print(c_string)
    print(f"{'Birthdates:':<20} {bmin}-{bmax}")
    print(f"{'Rarity:':<20} {cohort['outcome'].is_not_null().mean()*100:.2f}")
    return cohort

def save_cohort(cohort, task):
    cohort = cohort.select("person_id", "censor", "outcome", "predict")
    path = (FPATH.DATA / "destiny" / "outcomes" / task).with_suffix(".parquet")
    cohort.write_parquet(path)
    FPATH.copy_to_opposite_drive(path)

# %% [markdown]
# # Setup

# %% [markdown]
# ## Fullpop (+ lifelines and periods)

# %%
lifelines = pl.read_parquet(FPATH.NETWORK_DUMP_DIR / "lifelines_DUMP.parquet")
birthdays = (
    lifelines.sort("birthday").group_by("person_id").first()["person_id", "birthday"]
)

person_ids_too_short_seqs = pl.read_parquet(
    FPATH.NETWORK_DATA / "person_ids_seqlen_two_or_shorter.parquet"
)["person_id"]
fullpop = birthdays.filter(
    ~pl.col("person_id").is_in(person_ids_too_short_seqs)
).unique("person_id")
fullpop

# %%
fullpop.write_parquet(FPATH.NETWORK_DATA / "destiny" / "cohort" / "fullpop.parquet")
print(f"Saved fullpop.parquet with {len(fullpop):_} people")

# %% [markdown]
# #### Then adjust Lifelines for easier future

# %%
lifelines = lifelines.filter(pl.col("person_id").is_in(fullpop["person_id"]))
lifelines["person_id"].n_unique()

# %%
lifelines.write_parquet(FPATH.NETWORK_DATA / "destiny" / "cohort" / "lifelines.parquet")
print(f"Saved lifelines.parquet")

# %% [markdown]
# #### Then define the valid periods where fullpop resides in Denmark

# %%
adjusted_lifelines = lifelines.with_columns(event_start_date=
    pl.when(pl.col("event_cause_start").str.contains("1968"))
    .then(pl.col("birthday"))
    .otherwise(pl.col("event_start_date"))
)
periods = adjusted_lifelines.select("person_id", "event_start_date", "event_final_date")
print(periods["person_id"].n_unique())
periods

# %%
periods.write_parquet(FPATH.NETWORK_DATA / "destiny" / "cohort" / "periods.parquet")
print(f"Saved periods.parquet with {len(periods):_} observation periods")

# %% [markdown]
# ## Cohort definition

# %%
# Reload fullpop to ensure we have the saved version
fullpop = pl.read_parquet(FPATH.NETWORK_DATA / "destiny" / "cohort" / "fullpop.parquet")

# %%
train = fullpop.sample(fraction=0.7, with_replacement=False, shuffle=True, seed=73)
rest = fullpop.join(train, on="person_id", how="anti")
val = rest.sample(fraction=1/3, with_replacement=False, shuffle=True, seed=73)
test = rest.join(val, on="person_id", how="anti")
print(f"Split sizes: train={len(train):_}, val={len(val):_}, test={len(test):_}")

# %%
# Verify no overlap
assert len(train.join(val, on="person_id")) == 0, "Train/val overlap!"
assert len(train.join(test, on="person_id")) == 0, "Train/test overlap!"
assert len(val.join(test, on="person_id")) == 0, "Val/test overlap!"
print("✓ No overlap between splits")

# %%
train.write_parquet(FPATH.NETWORK_DATA / "destiny" / "cohort" / "train.parquet")
val.write_parquet(FPATH.NETWORK_DATA / "destiny" / "cohort" / "val.parquet")
test.write_parquet(FPATH.NETWORK_DATA / "destiny" / "cohort" / "test.parquet")
print("✓ Saved train/val/test splits")

# %% [markdown]
# # Death

# %%
name = "Death"
df = pl.read_parquet(FPATH.NETWORK_DATA / "destiny" / "cohort" / "lifelines.parquet").rename({"event_final_date": "date_col"})
print(f"***** {name} *****")
positives = get_positives(df, "event_cause_final", ["Doed"], fullpop)
cohort = get_cohort(fullpop, positives, periods)
save_cohort(cohort, f"{name}")

# %% [markdown]
# # Longevity

# %%
def get_longevity_cohort(fullpop, lifelines, periods, target_age=80, year_reduce=5):
    """
    Create a longevity outcome where:
    - Positives: People who reached target_age (either died at ≥target_age or alive and ≥target_age)
    - Negatives: People who died before target_age
    - Excluded: People alive but haven't reached target_age yet (ambiguous)
    """
    print(f"{'Creating longevity cohort for age:':<40} {target_age}")

    # Get death dates and birthdays
    deaths = lifelines.filter(
        pl.col("event_cause_final") == "Doed"
    ).select(["person_id", "event_final_date", "birthday"])

    # Calculate age at death for those who died
    deaths_with_age = deaths.with_columns(
        age_at_death = (pl.col("event_final_date").dt.year() - pl.col("birthday").dt.year())
    )

    # Join with fullpop to get everyone
    cohort = fullpop.join(
        deaths_with_age.select(["person_id", "event_final_date", "age_at_death"]),
        on="person_id",
        how="left"
    )

    # Calculate current age for those still alive (use 2024 as reference)
    import datetime
    current_year = 2024
    cohort = cohort.with_columns(
        current_age = pl.when(pl.col("age_at_death").is_null())
        .then(current_year - pl.col("birthday").dt.year())
        .otherwise(pl.col("age_at_death"))
    )

    # Classify into positives, negatives, and ambiguous
    cohort = cohort.with_columns(
        reached_target_age = pl.when(pl.col("current_age") >= target_age)
        .then(True)
        .when(pl.col("age_at_death").is_not_null() & (pl.col("age_at_death") < target_age))
        .then(False)
        .otherwise(None)  # Ambiguous: alive but haven't reached target_age yet
    )

    # Filter out ambiguous cases
    cohort = cohort.filter(pl.col("reached_target_age").is_not_null())

    print(f"{'Total people after filtering:':<40} {len(cohort):_}")
    print(f"{'Positives (reached {target_age}):':<40} {cohort['reached_target_age'].sum():_}")
    print(f"{'Negatives (died before {target_age}):':<40} {(~cohort['reached_target_age']).sum():_}")

    # For positives: outcome is the date they reached target_age
    # For negatives: outcome is null (they never reached it)
    cohort = cohort.with_columns(
        outcome = pl.when(pl.col("reached_target_age"))
        .then(pl.col("birthday") + pl.duration(days=int(target_age * 365.25)))
        .otherwise(None),
        censor = pl.when(pl.col("reached_target_age"))
        .then(pl.col("birthday") + pl.duration(days=int(target_age * 365.25)))
        .otherwise(pl.col("event_final_date"))
    )

    # Apply birth year cohort matching (like get_cohort does)
    positives_only = cohort.filter(pl.col("reached_target_age"))
    bmin, bmax = positives_only["birthday"].dt.year().min(), positives_only["birthday"].dt.year().max()
    cohort = cohort.filter(
        (pl.col("birthday").dt.year() >= bmin) & (pl.col("birthday").dt.year() <= bmax)
    )

    # For negatives: sample a "pseudo-censor" date from positives in same birth cohort
    # This ensures prediction points are realistic
    cohort = cohort.with_columns(
        year=pl.col("birthday").dt.year() // year_reduce
    ).group_by("year").map_groups(
        lambda group: group.with_columns(
            censor=pl.when(~pl.col("reached_target_age"))
            .then(
                pl.col("censor").filter(pl.col("reached_target_age"))
                .sample(pl.len(), with_replacement=True)
            )
            .otherwise(pl.col("censor"))
        )
    )

    # Filter to valid observation periods
    cohort = cohort.join(periods, on="person_id").filter(
        (pl.col("censor") >= pl.col("event_start_date")) &
        (pl.col("censor") <= pl.col("event_final_date"))
    ).drop("event_start_date", "event_final_date")

    # Create prediction points at meaningful ages (e.g., 50, 60, 70 to predict if they'll reach 80)
    prediction_ages = [target_age - 30, target_age - 20, target_age - 10, target_age - 5]
    cohort = cohort.with_columns(
        predict=pl.concat_list([
            pl.when(
                (pl.col("birthday") + pl.duration(days=int(age * 365.25))) < pl.col("censor")
            )
            .then(pl.col("birthday") + pl.duration(days=int(age * 365.25)))
            .otherwise(None)
            for age in prediction_ages
        ]).list.drop_nulls()
    ).filter(pl.col("predict").list.len() > 0)

    print(f"{'Final cohort after filtering:':<40} {len(cohort):_}")

    # Show train/val/test split stats
    c_string = f"{'Train/val/test:':<40} " + '  /  '.join(
        f"{len(subset):_} ({len(subset.filter(pl.col('outcome').is_not_null())):_})"
        for subset in [cohort.join(s, on="person_id") for s in [train, val, test]]
    )
    print(c_string)
    print(f"{'Birthdates:':<40} {bmin}-{bmax}")
    print(f"{'Rarity (reached {target_age}):':<40} {cohort['outcome'].is_not_null().mean()*100:.2f}%")

    return cohort.select("person_id", "censor", "outcome", "predict")

# %%
# Create Longevity-80 outcome
name = "Longevity-80"
lifelines = pl.read_parquet(FPATH.NETWORK_DATA / "destiny" / "cohort" / "lifelines.parquet")
print(f"***** {name} *****")
cohort = get_longevity_cohort(fullpop, lifelines, periods, target_age=80)
save_cohort(cohort, name)

# %%
# Optional: Create multiple longevity thresholds
# name = "Longevity-85"
# cohort = get_longevity_cohort(fullpop, lifelines, periods, target_age=85)
# save_cohort(cohort, name)

# name = "Longevity-90"
# cohort = get_longevity_cohort(fullpop, lifelines, periods, target_age=90)
# save_cohort(cohort, name)

# %% [markdown]
# # Health outcomes

# %%
lpr = pl.read_parquet(FPATH.NETWORK_DATA / "destiny" / "lpr.parquet", columns=["person_id", "date_col", "aktionsdiagnose_adaptrunc"])
lpr.head()

# %%
outcomes = {
    "depression": ["HEA_ICD10_DF32", "HEA_ICD10_DF33"], # Psych
    "schizophrenia": ["HEA_ICD10_DF20"], # Psych
    "type2-diabetes": ["HEA_ICD10_DE11"], # Chronic
    "osteonecrosis": ["HEA_ICD10_DM87"], # Chronic
    "colorectal-cancer": ["HEA_ICD10_DC18", "HEA_ICD10_DC20"], # Cancer
    "lung-cancer": ["HEA_ICD10_DC34"], # Cancer
    "arrhythmia": ["HEA_ICD10_DI47", "HEA_ICD10_DI49", "HEA_ICD10_DR00"], # Cardiovascular
    "stroke": ["HEA_ICD10_DI64"], # Cardiovascular
    "sleep-disorder": ["HEA_ICD10_DF51", "HEA_ICD10_DG47"], # Others
}
for name, codes in outcomes.items():
    print(f"***** {name} *****")
    positives = get_positives(lpr, "aktionsdiagnose_adaptrunc", codes, fullpop)
    cohort = get_cohort(fullpop, positives, periods)
    save_cohort(cohort, f"HEA_{name}")
    print()

# %% [markdown]
# # Education outcomes

# %%
graduation = pl.read_parquet(FPATH.NETWORK_DATA / "destiny" / "graduation.parquet")

# %%
edu = {
    "High-school": ["EDU_disced_20"],
    "Vocational": ["EDU_disced_30"],
    "Higher": ["EDU_disced_40", "EDU_disced_50", "EDU_disced_60", "EDU_disced_70"] 
}

for name, codes in edu.items():
    print(f"***** {name} *****")
    positives = get_positives(graduation, "disced", codes, fullpop)
    cohort = get_cohort(fullpop, positives, periods)
    save_cohort(cohort, f"EDU_{name}")
    print()

# %% [markdown]
# # Labor outcomes

# %%
name = "Unemployment"
df = pl.read_parquet(FPATH.NETWORK_DATA / "destiny" / "akm.parquet", columns=["person_id", "date_col", "socio13"])
print(f"***** {name} *****")
positives = get_positives(df, "socio13", ["LAB_socio13_210"], fullpop)
cohort = get_cohort(fullpop, positives, periods)
save_cohort(cohort, f"LAB_{name}")

# %%
name = "Millionare"
df = pl.read_parquet(FPATH.NETWORK_DUMP_DIR / "ind_DUMP_AUGMENTED.parquet", columns=["person_id", "referencetid", "assets", "source_table"]).filter(
    pl.col("assets") > 1_000_000
).rename({"referencetid": "date_col"})
print(f"***** {name} *****")
positives = get_positives(df, "source_table", [""], fullpop)
cohort = get_cohort(fullpop, positives, periods)
save_cohort(cohort, f"LAB_{name}")

# %%
name = "Disability-pension"
df = pl.read_parquet(FPATH.NETWORK_DATA / "destiny" / "akm.parquet", columns=["person_id", "date_col", "socio13"])
print(f"***** {name} *****")
positives = get_positives(df, "socio13", ["LAB_socio13_321"], fullpop)
cohort = get_cohort(fullpop, positives, periods)
save_cohort(cohort, f"LAB_{name}")

# %%
name = "Disability-pension2"
df= pl.read_parquet(FPATH.NETWORK_DATA / "destiny" / "amrun.parquet", columns=["person_id", "date_col", "soc_status_kode"])
print(f"***** {name} *****")
positives = get_positives(df, "soc_status_kode", ["LAB_soc_status_kode_411"], fullpop)
cohort = get_cohort(fullpop, positives, periods)
save_cohort(cohort, f"LAB_{name}")

# %% [markdown]
# # Demographic outcomes

# %%
name = "First-child"
df = pl.read_parquet(FPATH.NETWORK_DATA / "destiny" / "family_births_deaths.parquet").filter(
    pl.col("event").str.contains("Birth")
)
print(f"***** {name} *****")
positives = get_positives(df, "relation", ["DEM_relation_Child"], fullpop)
cohort = get_cohort(fullpop, positives, periods)
save_cohort(cohort, f"DEM_{name}")

# %%
name = "Marriage"
df = pl.read_parquet(FPATH.NETWORK_DATA / "destiny" / "marital_status.parquet")
print(f"***** {name} *****")
positives = get_positives(df, "civst", ["DEM_civst_G"], fullpop)
cohort = get_cohort(fullpop, positives, periods)
save_cohort(cohort, f"DEM_{name}")

# %%
name = "Migration"
df = pl.read_parquet(FPATH.NETWORK_DATA / "destiny" / "migration.parquet")
print(f"***** {name} *****")
positives = get_positives(df, "event", ["DEM_Udvandret"], fullpop)
cohort = get_cohort(fullpop, positives, periods)
save_cohort(cohort, f"DEM_{name}")

# %% [markdown]
# # Social conditions outcomes

# %%
name = "Drug-treatment"
df = pl.read_parquet(FPATH.NETWORK_DATA / "destiny" / "drug_treatment_start.parquet")
print(f"***** {name} *****")
positives = get_positives(df, "start_drug_service", ["SOC_start_drug_service_1"], fullpop)
cohort = get_cohort(fullpop, positives, periods)
save_cohort(cohort, f"SOC_{name}")

# %%
name = "Preventative-measures"
df = pl.read_parquet(FPATH.NETWORK_DATA / "destiny" / "preventative_measures_start.parquet")
print(f"***** {name} *****")
positives = get_positives(df, "start_precaution", ["SOC_start_precaution_1"], fullpop)
cohort = get_cohort(fullpop, positives, periods)
save_cohort(cohort, f"SOC_{name}")

# %%
name = "Penal-crime"
df = pl.read_parquet(FPATH.NETWORK_DATA / "destiny" / "decisions.parquet").filter(
    pl.col("afgtypko").is_in(
        [f"SOC_afgtypko_{i}" for i in [1, 2, 3, 4, 5, 6, 8, 12, 13, 15, 19, 20, 21, 81, 82, 83, 84, 86, 88, 14, 18, 70, 80, 60, 89, 90]]
    )
)
print(f"***** {name} *****")
positives = get_positives(df, "ger7_adaptrunc", ["SOC_ger7_1"], fullpop)
cohort = get_cohort(fullpop, positives, periods)
save_cohort(cohort, f"SOC_{name}")

# %% [markdown]
# # Data stats

# %%
lf = pl.scan_parquet(FPATH.NETWORK_DATA / "destiny" / "amrun.parquet")

# %%
for col in lf.columns:
    df = pl.read_parquet(FPATH.NETWORK_DATA / "destiny" / "amrun.parquet", columns=[col])
    for column in df.columns:
        if column  == "person_id":
            print(round(len(df) / 1_000_000, 2))
            print(column, round(df[column].n_unique() / 1_000_000, 2))
        elif column == "date_col":
            print(column, df[column].dt.year().min(), df[column].dt.year().max())
        else:
            print(column, round(df[column].is_not_null().sum() / 1_000_000, 2), df[column].filter(df[column].is_not_null()).n_unique(), df[column].unique().to_list()[:5])

# %%
# df = pl.read_parquet(FPATH.NETWORK_DATA / "destiny" / "akm.parquet", columns=["socio13","socio_gl"])
# df = pl.read_parquet(FPATH.NETWORK_DATA / "destiny" / "akm.parquet", columns=["disco08_alle_indk_13","disco08_loen_indk", "disco08_sel_indk", "disco_alle_indk_13", "discoloen_indk", "discosel_indk"])
df = pl.read_parquet(FPATH.NETWORK_DATA / "destiny" / "akm.parquet", columns=["nace_db07_13","nacea_db07","nacei_db07", "nace_13", "nacea", "nacei", "branche_77", "brchi", "brchl"])
count, unqs = 0, []
for col in df.columns:
    count += df[col].is_not_null().sum()
    unqs += df[col].filter(df[col].is_not_null()).unique().to_list()
print(round(count / 1_000_000, 2), len(set(unqs)))

# %% [markdown]
# # Outcome distributions

# %%
dir_path = FPATH.NETWORK_DATA / "destiny" / "outcomes"

fig, axs = plt.subplots(7, 3)
fig.set_size_inches(12, 20)
axs = axs.flatten()
for ax, outcome in zip(axs, dir_path.glob("*_*.parquet")):
    df = pl.read_parquet(outcome)
    df = df.filter(pl.col("outcome").is_not_null()).select(pl.col("outcome").dt.year())["outcome"].value_counts().sort("outcome")
    ax.plot(df["outcome"], (df["count"] / df["count"].sum())*100)
    ax.set_xlabel("Calendar year", fontsize=12)
    ax.set_ylabel("Percentage of positives", fontsize=12)
    ax.tick_params("x", labelsize=10)
    ax.tick_params("y", labelsize=10)
    outcome_name = outcome.stem.split("_")[-1].replace("-", " ").capitalize()
    if outcome_name[-1] == "2":
        print("Adjusted", outcome_name, outcome_name[:-1])
        outcome_name = outcome_name[:-1]
    ax.set_title(outcome_name, fontsize=14)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
plt.tight_layout()
plt.savefig(FPATH.FIGURES / "destiny" / "outcome_freq-outcome_year.png")
plt.show()

# %%
dir_path = FPATH.NETWORK_DATA / "destiny" / "outcomes"
info = pl.read_parquet(FPATH.NETWORK_DATA / "destiny" / "cohort" / "fullpop.parquet")


fig, axs = plt.subplots(7, 3)
fig.set_size_inches(12, 20)
axs = axs.flatten()
for ax, outcome in zip(axs, dir_path.glob("*_*.parquet")):
    df = pl.read_parquet(outcome)
    df = df.join(info, on="person_id")
    df = df.filter(pl.col("outcome").is_not_null()).select(pl.col("birthday").dt.year())["birthday"].value_counts().sort("birthday")
    ax.plot(df["birthday"], (df["count"] / df["count"].sum())*100)
    ax.set_xlabel("Birth year", fontsize=12)
    ax.set_ylabel("Percentage of positives", fontsize=12)
    ax.tick_params("x", labelsize=10)
    ax.tick_params("y", labelsize=10)
    outcome_name = outcome.stem.split("_")[-1].replace("-", " ").capitalize()
    if outcome_name[-1] == "2":
        print("Adjusted", outcome_name, outcome_name[:-1])
        outcome_name = outcome_name[:-1]
    ax.set_title(outcome_name, fontsize=14)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
plt.tight_layout()
plt.savefig(FPATH.FIGURES / "destiny" / "outcome_freq-birthyear.png")
plt.show()

# %%
dir_path = FPATH.NETWORK_DATA / "destiny" / "outcomes"
info = pl.read_parquet(FPATH.NETWORK_DATA / "destiny" / "cohort" / "fullpop.parquet")


fig, axs = plt.subplots(7, 3)
fig.set_size_inches(12, 20)
axs = axs.flatten()
for ax, outcome in zip(axs, dir_path.glob("*_*.parquet")):
    df = pl.read_parquet(outcome)
    df = df.join(info, on="person_id")
    df = df.filter(pl.col("outcome").is_not_null()).select(pl.col("outcome").dt.year() - pl.col("birthday").dt.year())["outcome"].value_counts().sort("outcome")
    ax.plot(df["outcome"], (df["count"] / df["count"].sum())*100)
    ax.set_xlabel("Age at outcome", fontsize=12)
    ax.set_ylabel("Percentage of positives", fontsize=12)
    ax.tick_params("x", labelsize=10)
    ax.tick_params("y", labelsize=10)
    outcome_name = outcome.stem.split("_")[-1].replace("-", " ").capitalize()
    if outcome_name[-1] == "2":
        print("Adjusted", outcome_name, outcome_name[:-1])
        outcome_name = outcome_name[:-1]
    ax.set_title(outcome_name, fontsize=14)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
plt.tight_layout()
plt.savefig(FPATH.FIGURES / "destiny" / "outcome_freq-age.png")
plt.show()

# %%



