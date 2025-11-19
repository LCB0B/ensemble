# %%
import polars as pl
from src.paths import FPATH

# %% [markdown]
# # Static time tokens

# %%
lifelines = pl.read_parquet(
    FPATH.NETWORK_DATA / "destiny" / "cohort" / "lifelines.parquet", 
    columns=["person_id", "birthday", "event_final_date"]
).sort("event_final_date")

# %%
start_end_df = lifelines.group_by("person_id").last().sort("event_final_date")

# %% [markdown]
# ### Birthday (age) tokens

# %%
birthday_df = start_end_df.with_columns(
    date_col=pl.date_ranges(
        pl.col("birthday").dt.offset_by("1y"), # Only start with AGE_1
        pl.col("event_final_date"),
        "1y"
    ),
).with_columns(
    age=pl.int_ranges(1, pl.col("date_col").list.len()+1, 1)
).drop("birthday", "event_final_date")

# %%
att_birthday = birthday_df.explode("date_col", "age").with_columns(
    age=pl.format("ATT_age{}", pl.col("age"))
)
att_birthday

# %%
att_birthday.write_parquet(FPATH.NETWORK_DATA / "destiny" / "att_birthday.parquet")

# %% [markdown]
# ### Calendar year tokens

# %%
calendar_df = start_end_df.with_columns(
    pl.col("birthday").dt.offset_by("1y").dt.truncate("1y"),
).with_columns(
    date_col=pl.date_ranges(
        pl.col("birthday"), pl.col("event_final_date"), "1y"
    ),
    calendar_years=pl.int_ranges(
        pl.col("birthday").dt.year(), pl.col("event_final_date").dt.year() + 1
    )
).drop("birthday", "event_final_date")

# %%
att_calendar = calendar_df.explode("date_col", "calendar_years").with_columns(
    calendar_years=pl.format("ATT_year{}", pl.col("calendar_years"))
)
att_calendar

# %%
att_calendar.write_parquet(FPATH.NETWORK_DATA / "destiny" / "att_calendar.parquet")

# %% [markdown]
# # Time tokens between events

# %%
lpr = pl.read_parquet(FPATH.NETWORK_DATA / "destiny" / "lpr.parquet")#.head(10_000_000)#.sort("date_col")#.head(1_000_000)

# %%
lpr1 = lpr.sort("date_col", maintain_order=True).with_columns(
    diff=pl.col("date_col").diff().over("person_id")
)
lpr2 = lpr.sort("person_id", "date_col", maintain_order=True).with_columns(
    diff=pl.col("date_col").diff().over("person_id")
)

# %%
lpr1.filter(pl.col("person_id") == 147267591)

# %%
lpr1.sort("person_id", "date_col").equals(lpr2)

# %%
lpr1.sort("person_id", "date_col").equals(lpr2.sort("person_id", "date_col"))

# %%
lpr1_sort = lpr1.sort("person_id", "date_col")

# %%
n = 10
lpr1_sort.tail(n).equals(lpr2.tail(n))

# %%
lpr1_sort.tail(11)

# %%
lpr2.tail(11)

# %%
edges = (
    [hour for hour in range(1, 24)] +
    [day*24 for day in range (1, 30)] +
    [month*24*30 for month in range(1, 13)] +
    [year*24*365 for year in range(1, 101)]
)
labels = (
    [f"[ATT_<{hour}HOUR]" for hour in range(1, 24)] +
    [f"[ATT_<{day}DAY]" for day in range (1, 30)] +
    [f"[ATT_<{month}MONTH]" for month in range(1, 13)] +
    [f"[ATT_<{year}YEAR]" for year in range(1, 101)]
    + ["[ATT_UNKNOWN]"]
)

# %%
foo = lpr.with_columns(
    diff2=pl.col("diff").dt.total_hours()
).with_columns(
    pl.col("diff2").cut(breaks=edges, labels=labels, left_closed=True)
)

# %%



