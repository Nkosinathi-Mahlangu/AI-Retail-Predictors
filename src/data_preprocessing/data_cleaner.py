"""
# ============================================================
# FILE OVERVIEW
# ============================================================
#
# This is src/data_cleaner.py — the first step in the AI Retail
# Predictors pipeline. Every other part of the project (model,
# chatbot, visualiser) depends on the data this file produces.
# It's the "front door": nothing useful happens until the raw
# CSV is loaded and cleaned.
#
# THE PROBLEM
# ------------
# The raw Kaggle dataset is generic retail data (categories like
# "Electronics", "Furniture") which don't mean anything to a
# filling-station manager. It also has missing values, duplicate
# rows, and messy formatting that would break everything
# downstream if left as-is.
#
# WHAT THIS FILE DOES
# ---------------------
# Loads the CSV, checks it's valid, cleans it up, and returns one
# trustworthy DataFrame the rest of the project can use.
#
# THE 6 ACCEPTANCE CRITERIA THIS FILE COVERS
# --------------------------------------------
# AC1 - Load the CSV, turn the Date column into real dates,
#       and error out clearly if the file is missing or broken.
# AC2 - Find rows missing required values, drop them, log how
#       many per column. If over 20% of rows are affected, stop
#       — the file is too messed up to trust.
# AC3 - Find exact duplicate rows (same Date + Store ID +
#       Product ID), keep only the first, log how many removed.
# AC4 - Rename generic categories to filling-station ones (e.g.
#       "Groceries" -> "Snacks") using a config file. Error out
#       if that config is missing.
# AC5 - End up with a clean DataFrame: all required columns
#       present, no nulls, no duplicate keys, sorted by date.
# AC6 - Print a plain-text summary of everything that was
#       cleaned, and optionally save it to a file (max 10,000
#       characters).
#
# HOW IT'S ORGANISED
# --------------------
# One public function, load_and_clean(), is all other modules
# need to call. Internally it runs six private helpers in order,
# one per acceptance criterion, so each piece stays simple and
# easy to test.
#
# WHAT'S COMING UP
# ------------------
# 1. Imports
# 2. Logging setup
# 3. Constants (column names etc.)
# 4. The public load_and_clean() function
# 5. The six private helpers, in the order they run
# 6. A command-line entry point
"""


# ============================================================
# SECTION: Imports
# ============================================================
#
# Bringing in the libraries this file needs.
#
# - json     : reads the category_mapping.json config file.
# - logging  : keeps a timestamped log of what's happening —
#              better than print() for real projects.
# - os       : general OS utilities (Path below covers most of
#              the file-path work).
# - sys      : lets us read command-line arguments and exit
#              with an error code.
# - pathlib.Path : a safer, cross-platform way to handle file
#              paths than joining strings manually.
# - pandas   : the main library for working with tables of data
#              (a DataFrame = one row per sale, one column per
#              field).

import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# SECTION: Logger Setup
# ============================================================
#
# With ~76,000 rows moving through this pipeline, a lot happens
# behind the scenes. Logging gives us a timestamped record of
# key events — files loaded, rows dropped, categories renamed —
# without cluttering the normal output.
#
# logging.basicConfig() sets up the default logger to show INFO
# level and above. getLogger("data_cleaner") gives this module
# its own named logger, so its messages are clearly labeled
# when other files (like main.py) import it.
#
# Example log line:
#   2026-09-13 22:29:23 [INFO] data_cleaner: Loaded 76000 rows...

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("data_cleaner")


# ============================================================
# SECTION: Constants
# ============================================================
#
# Fixed values shared across the whole file.
#
# Why bother? Column names like "Store ID" show up in many
# places. Define them once here, and if the CSV column ever
# changes name, we only fix it in one spot instead of ten.
#
# - REQUIRED_COLUMNS: the 14 columns every cleaned row must
#   have. Rows missing any of these get dropped. Note: "Demand"
#   is an extra column in the raw CSV — we keep it but don't
#   require it, since it's not in the project spec.
# - DATE_COLUMN: the date field's name. Kept separate since it
#   also becomes the DataFrame's index later.
# - DUPLICATE_KEY_COLUMNS: the three columns that together
#   should uniquely identify one sale.
# - _DEFAULT_CONFIG_PATH: where we expect to find the category
#   mapping file, built from this script's own location so it
#   works no matter where you run it from. The underscore
#   prefix marks it as "internal use only".

REQUIRED_COLUMNS = [
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Inventory Level",
    "Units Sold",
    "Units Ordered",
    "Price",
    "Discount",
    "Weather Condition",
    "Promotion",
    "Competitor Pricing",
    "Seasonality",
    "Epidemic",
]

DATE_COLUMN = "Date"
DUPLICATE_KEY_COLUMNS = [DATE_COLUMN, "Store ID", "Product ID"]

# Build the default config path relative to this file:
# parent -> data_preprocessing/, parent again -> src/,
# parent again -> project root, then into config/category_mapping.json.
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "category_mapping.json"


# ============================================================
# SECTION: Public API — load_and_clean()
# ============================================================
#
# This is the only function other modules should ever call.
# The idea: give it a CSV path, get back a clean DataFrame. The
# caller doesn't need to know about the six cleaning steps
# happening under the hood — that's the point of hiding
# complexity behind a simple interface.
#
# It works as a pipeline: each helper's output feeds into the
# next one.
#
#   _load_csv()               -> raw DataFrame
#   _load_category_mapping()  -> config (loaded early to fail fast)
#   _remove_missing_values()  -> DataFrame with no null rows
#   _drop_duplicates()        -> DataFrame with no duplicate keys
#   _relabel_categories()     -> filling-station category names
#   sort + set_index          -> datetime index, oldest first
#   _write_report()           -> prints/saves the summary
#
# NOTE ON ORDER: we load the category config BEFORE removing
# missing values on purpose. If the config is missing, we want
# to fail before touching any data at all, as required.
#
# Next: the six private helpers, in the order they're called.

def load_and_clean(
    csv_path: str,
    config_path: str = str(_DEFAULT_CONFIG_PATH),
    report_path: str = None,
) -> pd.DataFrame:
    """
    Load, validate, and clean the raw retail CSV dataset.

    Parameters
    ----------
    csv_path : str
        Path to the raw CSV file to ingest.
    config_path : str, optional
        Path to the category_mapping.json configuration file.
        Defaults to config/category_mapping.json relative to the project root.
    report_path : str, optional
        If provided, the cleaning summary report is also written to this file
        path in addition to stdout.  Pass None to skip file output.

    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame with:
        - All required columns present and non-null.
        - No duplicate (Date, Store ID, Product ID) combinations.
        - A datetime64 index (Date) sorted in ascending chronological order.
        - Category values relabelled to filling-station equivalents.

    Raises
    ------
    FileNotFoundError
        If csv_path does not exist.
    ValueError
        If the file cannot be parsed as CSV, missing-value rate exceeds 20%,
        or the category-mapping config is absent/unreadable.
    """
    # Step 1 (AC1): load the raw CSV. Keep the original row count —
    # we'll need it later for the report and the 20% threshold check.
    df, original_row_count = _load_csv(csv_path)

    # Step 2 (AC4 pre-check): load the category config now, before
    # touching any data, so a missing config fails immediately.
    category_mapping = _load_category_mapping(config_path)

    # Step 3 (AC2): drop rows with missing required values.
    df, missing_counts = _remove_missing_values(df, original_row_count)

    # Step 4 (AC3): drop duplicate rows.
    df, duplicate_count = _drop_duplicates(df)

    # Step 5 (AC4 apply): rename categories to filling-station names.
    df, relabel_counts, filter_count = _relabel_categories(df, category_mapping)

    # Step 6 (AC5): sort by date and make it the index. The extra
    # pd.to_datetime() call is just a safety net — pandas can
    # occasionally change the dtype after set_index.
    df = df.sort_values(DATE_COLUMN).set_index(DATE_COLUMN)
    df.index = pd.to_datetime(df.index)  # guarantee datetime64 index

    # Step 7 (AC6): build and print the summary report.
    cleaned_row_count = len(df)
    _write_report(
        original_row_count=original_row_count,
        cleaned_row_count=cleaned_row_count,
        missing_counts=missing_counts,
        duplicate_count=duplicate_count,
        relabel_counts=relabel_counts,
        filter_count=filter_count,
        report_path=report_path,
    )

    # Hand back the clean, trustworthy DataFrame.
    return df


# ============================================================
# SECTION: Private Helpers
# ============================================================
#
# The six private functions load_and_clean() relies on.
#
# Why split things up like this?
#   1. Readability  — each function's name says what it does.
#   2. Testability  — you can test _drop_duplicates() on its own.
#   3. Maintainability — need to change dedup logic? Edit one
#      small function, not a giant block of code.
#
# The leading underscore (_load_csv, etc.) is Python's way of
# saying "internal only, don't call this from outside."
#
# Implemented below in the order they run.


# ============================================================
# SECTION: Private Helper — _load_csv  (AC1)
# ============================================================
#
# Step one: get the file into memory. If this fails, nothing
# else matters, so we fail fast with a clear error message.
#
# What it does:
# 1. Turn the path into a Path object for safe file handling.
# 2. Check the file exists — raise FileNotFoundError if not.
# 3. Read the CSV, parsing the Date column into real dates.
# 4. Double-check the Date column actually became datetime
#    (sometimes pandas silently leaves oddly formatted dates
#    as plain strings).
# 5. Return the DataFrame plus the row count, so we always know
#    the original size for later comparisons.

def _load_csv(csv_path: str):
    """
    Load the raw CSV file and parse the Date column as datetime.

    Parameters
    ----------
    csv_path : str
        Path to the raw CSV file.

    Returns
    -------
    tuple[pd.DataFrame, int]
        The loaded DataFrame and the original row count.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at csv_path.
    ValueError
        If the file cannot be parsed as a valid CSV.
    """
    path = Path(csv_path)

    # Fail immediately if the file isn't there — include the bad
    # path so the user knows exactly what to check.
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: '{csv_path}'. "
            "Please verify the path and try again."
        )

    # Catch low-level parsing errors (wrong encoding, binary file,
    # etc.) and re-raise them as a friendlier ValueError.
    try:
        df = pd.read_csv(path, parse_dates=[DATE_COLUMN])
    except Exception as exc:
        raise ValueError(
            f"Cannot parse '{csv_path}' as a CSV file. Reason: {exc}"
        ) from exc

    # parse_dates is "best effort" — make sure it actually worked.
    if not pd.api.types.is_datetime64_any_dtype(df[DATE_COLUMN]):
        raise ValueError(
            f"The '{DATE_COLUMN}' column in '{csv_path}' could not be "
            "converted to datetime64. Check the date format."
        )

    original_row_count = len(df)
    logger.info("Loaded %d rows from '%s'.", original_row_count, csv_path)
    return df, original_row_count


# ============================================================
# SECTION: Private Helper — _load_category_mapping  (AC4 pre)
# ============================================================
#
# Loads the config that tells us how to rename generic retail
# categories into filling-station ones.
#
# Why a separate config file instead of hard-coding the
# mappings? So a non-programmer can update the rules without
# touching code, and tests can swap in different configs to
# check edge cases.
#
# We load this early — before any data changes happen — because
# if the config is missing, we want to fail cleanly before we've
# modified anything.
#
# What it does:
# 1. Check the config file exists — raise ValueError if not
#    (this is a config problem, not a missing-data problem).
# 2. Parse the JSON.
# 3. Confirm it has the two required keys: "mappings" and
#    "valid_target_categories".
# 4. Return the whole config so _relabel_categories() can use it.

def _load_category_mapping(config_path: str) -> dict:
    """
    Load and validate the category-mapping configuration file.

    Parameters
    ----------
    config_path : str
        Path to the category_mapping.json file.

    Returns
    -------
    dict
        A dict with keys 'mappings' and 'valid_target_categories'.

    Raises
    ------
    ValueError
        If the config file is absent, unreadable, or missing required keys.
    """
    path = Path(config_path)

    # Fail before touching any data if the config isn't there.
    if not path.exists():
        raise ValueError(
            f"Category-mapping configuration not found: '{config_path}'. "
            "Cannot proceed without it."
        )

    # Catch malformed JSON and re-raise with a clearer message.
    try:
        with open(path, encoding="utf-8") as fh:
            config = json.load(fh)
    except Exception as exc:
        raise ValueError(
            f"Cannot read category-mapping config '{config_path}'. Reason: {exc}"
        ) from exc

    # Make sure both expected keys are present — otherwise
    # _relabel_categories() would crash later with a confusing error.
    if "mappings" not in config or "valid_target_categories" not in config:
        raise ValueError(
            f"Category-mapping config '{config_path}' is missing required keys "
            "'mappings' and/or 'valid_target_categories'."
        )

    logger.info(
        "Loaded category mapping from '%s': %d rules.", config_path, len(config["mappings"])
    )
    return config


# ============================================================
# SECTION: Private Helper — _remove_missing_values  (AC2)
# ============================================================
#
# Scans for missing values in required columns and drops those
# rows.
#
# Why this matters: models can't handle nulls. A missing price,
# for example, would crash a regression or quietly produce
# wrong predictions. Removing these rows early means everything
# downstream gets complete data.
#
# Why the 20% cutoff: dropping a few rows is normal. But if 30
# or 40% of the data is missing something, that's a sign the
# wrong file was used or the export was broken — silently
# dropping a third of the dataset would badly bias any model
# trained on it. The threshold forces a human to investigate.
#
# What it does:
# 1. Only check required columns that actually exist in the
#    DataFrame (in case the CSV is missing a column entirely).
# 2. Count nulls per column, for the report.
# 3. Work out what % of all rows have at least one missing
#    required value.
# 4. If that's over 20%, raise an error.
# 5. Drop the affected rows.
# 6. Return the cleaned DataFrame plus the per-column counts.
#
# Note: .isna().any(axis=1) checks each row across the given
# columns and returns True if any of them is null.

def _remove_missing_values(df: pd.DataFrame, original_row_count: int):
    """
    Remove rows with missing values in any required column and log counts.

    Parameters
    ----------
    df : pd.DataFrame
        The raw loaded DataFrame.
    original_row_count : int
        Number of rows before any cleaning (used for the 20% threshold check).

    Returns
    -------
    tuple[pd.DataFrame, dict]
        The DataFrame after missing-value removal, and a dict mapping each
        column name to the count of rows dropped for that column.

    Raises
    ------
    ValueError
        If more than 20% of rows across all required columns contain missing values.
    """
    # Only check required columns that are actually present in the CSV.
    cols_to_check = [c for c in REQUIRED_COLUMNS if c in df.columns]
    missing_counts = {}

    # Count nulls per column before dropping anything, so the report
    # reflects the original state.
    for col in cols_to_check:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            missing_counts[col] = int(n_missing)
            logger.info("Column '%s': %d missing value(s) found.", col, n_missing)

    # How many rows have at least one missing required value?
    rows_with_any_missing = df[cols_to_check].isna().any(axis=1).sum()

    # Measure against the ORIGINAL row count, not the current size.
    missing_pct = rows_with_any_missing / original_row_count if original_row_count > 0 else 0

    # AC2 safeguard: stop if too much data is missing.
    if missing_pct > 0.20:
        raise ValueError(
            f"Missing-value rate is {missing_pct:.1%} ({rows_with_any_missing} rows), "
            "which exceeds the 20% threshold. Cannot produce a cleaned DataFrame."
        )

    # Drop rows missing a value in any required column. Nulls in
    # extra columns (like "Demand") are left alone.
    before = len(df)
    df = df.dropna(subset=cols_to_check)
    after = len(df)
    total_dropped = before - after

    if total_dropped > 0:
        logger.info("Removed %d row(s) due to missing required values.", total_dropped)
    else:
        logger.info("No missing values found in required columns.")

    return df, missing_counts


# ============================================================
# SECTION: Private Helper — _drop_duplicates  (AC3)
# ============================================================
#
# Removes rows that represent the same sale recorded more than
# once.
#
# What counts as a duplicate: two rows sharing the exact same
# Date, Store ID, and Product ID. That combination should only
# ever appear once — if it shows up twice, it's a data-entry
# mistake or an export glitch.
#
# Why keep the first occurrence: without more context we can't
# tell which copy is "right". Keeping the first is simple and
# consistent — and if the data's already sorted chronologically,
# "first" usually means the earliest, most reliable record.
#
# drop_duplicates(subset=..., keep="first") does the work:
# subset defines which columns count toward "identical", and
# keep="first" keeps only the earliest of each group.

def _drop_duplicates(df: pd.DataFrame):
    """
    Drop duplicate rows sharing the same (Date, Store ID, Product ID) key,
    keeping the first occurrence.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame after missing-value handling.

    Returns
    -------
    tuple[pd.DataFrame, int]
        The deduplicated DataFrame and the count of rows removed.
    """
    before = len(df)

    # Drop rows where the key columns exactly match an earlier row.
    df = df.drop_duplicates(subset=DUPLICATE_KEY_COLUMNS, keep="first")

    after = len(df)
    duplicate_count = before - after

    if duplicate_count > 0:
        logger.info(
            "Removed %d duplicate row(s) based on (Date, Store ID, Product ID).",
            duplicate_count,
        )
    else:
        logger.info("No duplicate (Date, Store ID, Product ID) records found.")

    return df, duplicate_count


# ============================================================
# SECTION: Private Helper — _relabel_categories  (AC4 apply)
# ============================================================
#
# Translates the dataset's generic retail categories into names
# that make sense for a filling station.
#
# Why: "Electronics" and "Furniture" mean nothing to someone
# stocking Snacks, Beverages, Airtime, and Motor Accessories.
#
# Why it's driven by a config file, not hard-coded: a
# non-programmer can update the mappings, and tests can swap in
# a different config to check edge cases.
#
# Two phases:
#
# Phase 1 — Relabel: for each (old_name -> new_name) pair in the
# config, find every row where Category equals old_name and
# overwrite it with new_name. Log how many rows changed.
#
# Phase 2 — Filter: after relabelling, drop any row whose
# Category still isn't one of the valid target categories —
# this catches categories the config didn't account for at all.
#
# Note on boolean masks: `mask = df["Category"] == "Groceries"`
# gives a True/False value per row. `df.loc[mask, "Category"] =
# "Snacks"` then updates only the matching rows — "for every row
# where Category is Groceries, change it to Snacks."

def _relabel_categories(df: pd.DataFrame, config: dict):
    """
    Relabel Category values to filling-station equivalents using the mapping
    config, and filter out any rows whose Category is not in the valid targets.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame after duplicate removal.
    config : dict
        The loaded category mapping configuration.

    Returns
    -------
    tuple[pd.DataFrame, dict, int]
        The relabelled DataFrame, a dict of {original_cat: (new_cat, count)},
        and the count of rows filtered out for having no valid mapping.
    """
    mappings: dict = config["mappings"]
    valid_targets: list = config["valid_target_categories"]

    # Records every relabelling decision, e.g. {"Groceries": ("Snacks", 30400)}.
    relabel_counts = {}

    # --- Phase 1: Relabel ---
    for original_cat, new_cat in mappings.items():
        mask = df["Category"] == original_cat
        count = int(mask.sum())  # plain Python int, safer for logging/JSON

        if count > 0:
            df.loc[mask, "Category"] = new_cat
            relabel_counts[original_cat] = (new_cat, count)
            logger.info(
                "Relabelled %d row(s): '%s' -> '%s'.", count, original_cat, new_cat
            )

    # --- Phase 2: Filter ---
    # Drop rows whose Category still isn't a recognised, valid target.
    before = len(df)
    df = df[df["Category"].isin(valid_targets)]
    after = len(df)
    filter_count = before - after

    if filter_count > 0:
        logger.info(
            "Filtered out %d row(s) with unrecognised Category values.", filter_count
        )
    else:
        logger.info("All Category values mapped successfully; no rows filtered out.")

    return df, relabel_counts, filter_count


# ============================================================
# SECTION: Private Helper — _write_report  (AC6)
# ============================================================
#
# Builds the human-readable summary of everything the cleaner
# did.
#
# Why it matters: cleaned data looks a lot like raw data at a
# glance. Without a report, nobody would know that 12,160 rows
# got relabelled or that zero duplicates were found. This is the
# audit trail.
#
# The 10,000-character cap keeps the report from ballooning on
# very wide datasets — we truncate and add a notice if needed.
#
# What it does:
# 1. Build the report as a list of lines, then join them at the
#    end (cheaper than repeatedly gluing strings together).
# 2. Fill in each section, with a friendly fallback message if
#    there's nothing to report (e.g. "No duplicates found").
# 3. Truncate if it's over 10,000 characters.
# 4. Always print to the terminal.
# 5. Optionally save to a file too, creating the folder if it
#    doesn't already exist.

def _write_report(
    original_row_count: int,
    cleaned_row_count: int,
    missing_counts: dict,
    duplicate_count: int,
    relabel_counts: dict,
    filter_count: int,
    report_path: str = None,
) -> None:
    """
    Produce a plain-text summary report of all cleaning actions.

    The report is always printed to stdout and, if report_path is provided,
    also saved as a plain-text file.  The report will not exceed 10 000 characters.

    Parameters
    ----------
    original_row_count : int
        Number of rows in the raw CSV.
    cleaned_row_count : int
        Number of rows in the final cleaned DataFrame.
    missing_counts : dict
        Mapping of column name -> count of rows dropped for missing values.
    duplicate_count : int
        Number of duplicate rows removed.
    relabel_counts : dict
        Mapping of original_category -> (new_category, row_count) for relabels.
    filter_count : int
        Number of rows removed because their Category was not in valid targets.
    report_path : str, optional
        File path to write the report.  None means stdout only.

    Returns None
    """
    # Build line by line, join once at the end.
    lines = [
        "=" * 60,
        "DATA CLEANING SUMMARY REPORT",
        "=" * 60,
        f"Original row count    : {original_row_count:,}",   # :, adds thousands separator
        f"Cleaned row count     : {cleaned_row_count:,}",
        f"Total rows removed    : {original_row_count - cleaned_row_count:,}",
        "",
        "--- Missing Value Removal ---",
    ]

    # Say explicitly when nothing was found, instead of leaving a
    # blank, confusing section.
    if missing_counts:
        for col, count in missing_counts.items():
            lines.append(f"  {col:<30}: {count:,} row(s) affected")  # :<30 left-aligns in 30 chars
    else:
        lines.append("  No missing values detected in required columns.")

    lines += [
        "",
        "--- Duplicate Removal ---",
        f"  Rows removed (same Date + Store ID + Product ID): {duplicate_count:,}",
        "",
        "--- Category Relabelling ---",
    ]

    if relabel_counts:
        for original_cat, (new_cat, count) in relabel_counts.items():
            lines.append(f"  '{original_cat}' -> '{new_cat}': {count:,} row(s)")
    else:
        lines.append("  No category relabelling applied.")

    lines += [
        "",
        "--- Category Filtering ---",
        f"  Rows removed (unrecognised Category after mapping): {filter_count:,}",
        "=" * 60,
    ]

    report = "\n".join(lines)

    # Enforce the 10,000-character cap (truncate at 9,990 to leave
    # room for the notice).
    if len(report) > 10_000:
        report = report[:9_990] + "\n[... report truncated to 10 000 characters ...]"

    # Always show it in the terminal.
    print(report)

    # Optionally save it to a file too, creating the folder if needed.
    if report_path:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as fh:
            fh.write(report)
        logger.info("Summary report saved to '%s'.", report_path)


# ============================================================
# SECTION: CLI Entry Point
# ============================================================
#
# This runs when someone executes the file directly, e.g.:
#
#     python src/data_preprocessing/data_cleaner.py data/raw/sales_data.csv
#
# Why the `if __name__ == "__main__":` guard: a Python file can
# be imported OR run directly. This guard means the CLI code
# only runs when the file itself is executed — if another module
# does `from data_cleaner import load_and_clean`, this block is
# skipped, so importing the file doesn't accidentally try to
# read command-line arguments and crash.
#
# How arguments work: sys.argv[0] is the script name, and
# sys.argv[1:] are whatever the user typed after it — csv_path
# (required), then optional config_path and report_path. We use
# a simple `value if len(args) > n else default` pattern for the
# optional ones.
#
# Error handling: we only catch FileNotFoundError and ValueError
# — the two errors load_and_clean() can raise on purpose.
# Anything else (like an out-of-memory error) is left to
# propagate normally, so the real traceback is visible instead
# of being hidden.

if __name__ == "__main__":
    # Drop the script filename, keep only the user-supplied args.
    args = sys.argv[1:]

    # No arguments? Show usage and quit with an error code.
    if not args:
        print(
            "Usage: python src/data_preprocessing/data_cleaner.py data/raw/sales_data.csv [config_path] [report_path]",
            file=sys.stderr,
        )
        sys.exit(1)

    # Required arg first, optional args fall back to defaults.
    _csv = args[0]
    _config = args[1] if len(args) > 1 else str(_DEFAULT_CONFIG_PATH)
    _report = (
        args[2]
        if len(args) > 2
        else str(
            Path(__file__).resolve().parent.parent.parent / "outputs" / "cleaning_report.txt"
        )
    )

    try:
        cleaned_df = load_and_clean(_csv, _config, _report)

        print(f"\nCleaning complete. DataFrame shape: {cleaned_df.shape}")
        print(f"Date index range: {cleaned_df.index.min()} to {cleaned_df.index.max()}")
        print(f"Categories present: {sorted(cleaned_df['Category'].unique())}")

    except (FileNotFoundError, ValueError) as err:
        # Show a clean error message, not a full traceback —
        # friendlier for whoever's running this from the terminal.
        logger.error("Cleaning failed: %s", err)
        sys.exit(1)


# ============================================================
# END-OF-FILE SUMMARY
# ============================================================
#
# WHAT THIS FILE DOES
# ---------------------
# data_cleaner.py is the first and most important module in the
# AI Retail Predictors project. It turns a raw, generic Kaggle
# CSV into one clean, trustworthy DataFrame that every other
# module can rely on.
#
# MAIN PIECES
# ------------
# 1. Imports          — json, logging, sys, pathlib, pandas.
# 2. Logger setup      — a named "data_cleaner" logger.
# 3. Constants          — REQUIRED_COLUMNS, DATE_COLUMN,
#    DUPLICATE_KEY_COLUMNS, defined once and reused everywhere.
# 4. load_and_clean()   — the single entry point; runs the six
#    steps in order and returns the cleaned DataFrame.
# 5. _load_csv()               [AC1] loads + parses dates.
# 6. _load_category_mapping()  [AC4 pre] loads config early.
# 7. _remove_missing_values()  [AC2] drops nulls, enforces 20% cap.
# 8. _drop_duplicates()        [AC3] removes repeated sales records.
# 9. _relabel_categories()     [AC4 apply] renames + filters categories.
# 10. _write_report()          [AC6] builds the audit trail.
# 11. CLI entry point           — lets the file be run standalone.
#
# HOW IT ALL FLOWS
# ------------------
#   Raw CSV
#     -> _load_csv()               : DataFrame + original count
#     -> _load_category_mapping()  : config dict (fail-fast)
#     -> _remove_missing_values()  : trimmed DataFrame
#     -> _drop_duplicates()        : deduplicated DataFrame
#     -> _relabel_categories()     : filling-station categories
#     -> sort + set_index          : datetime index, oldest first
#     -> _write_report()           : printed + saved summary
#     -> return cleaned DataFrame
#
# WHERE THINGS STAND
# --------------------
# The pipeline can now take the raw sales_data.csv and produce a
# clean DataFrame with:
#   - 76,000 rows (nothing lost from this particular dataset)
#   - a datetime index sorted oldest to newest
#   - 4 filling-station categories: Airtime, Beverages,
#     Motor Accessories, Snacks
#   - no missing values in any required column
#   - no duplicate (Date, Store ID, Product ID) records
#
# WHAT'S NEXT
# -------------
# 1. Chronological train/test split (80/20 by date, no future
#    leakage into training).
# 2. Time-series decomposition using statsmodels'
#    seasonal_decompose on the training set.
# 3. Train the ML/DL models (Random Forest / Gradient Boosting
#    and an LSTM/Dense network).
#
# Next file to build: src/time_series_analyser.py
# ============================================================
