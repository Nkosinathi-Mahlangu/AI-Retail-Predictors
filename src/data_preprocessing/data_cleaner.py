"""
# ============================================================
# FILE OVERVIEW
# ============================================================
#
# We are currently in: src/data_cleaner.py
#
# WHERE THIS FILE FITS IN THE PROJECT
# ------------------------------------
# This is the very first module in the AI Retail Predictors
# pipeline. Every other module (the ML model, the chatbot,
# the visualiser) depends on the data that this file produces.
# Think of it as the front door of the entire system — nothing
# useful happens until the raw CSV has been loaded and cleaned.
#
# WHAT PROBLEM THIS FILE SOLVES
# ------------------------------
# The raw Kaggle dataset is "general retail" data. It has
# category names like "Electronics" and "Furniture" which mean
# nothing to a filling-station manager. It may also contain
# missing values, duplicate records, and inconsistent formats
# that would silently corrupt every downstream calculation if
# left untreated. This file fixes all of that before the data
# is handed to anyone else.
#
# MAIN RESPONSIBILITY
# -------------------
# Load the raw CSV, validate its structure, clean its contents,
# and return a single trusted pandas DataFrame that the rest
# of the pipeline can use with confidence.
#
# WHAT THIS FILE IMPLEMENTS (the 6 Acceptance Criteria)
# ------------------------------------------------------
# AC1 - Load the CSV file and parse the Date column as a
#       proper datetime type. Raise a clear error if the file
#       is missing or unreadable.
# AC2 - Find rows that have missing values in any required
#       column. Remove those rows and log how many were removed
#       per column. If more than 20% of all rows are affected,
#       stop and raise an error — the dataset is too corrupted
#       to be useful.
# AC3 - Find rows that are exact duplicates by the combination
#       of (Date, Store ID, Product ID). Keep only the first
#       occurrence and log how many were removed.
# AC4 - Translate generic retail category names into filling-
#       station equivalents (e.g. "Groceries" -> "Snacks") using
#       a separate config file. Raise an error if that config
#       file is missing.
# AC5 - Produce the final cleaned DataFrame: all required
#       columns present, no nulls, no duplicate keys, and a
#       datetime index sorted oldest-to-newest.
# AC6 - Write a plain-text summary report listing every
#       cleaning action taken. Print it to the terminal and
#       optionally save it to a file (max 10 000 characters).
#
# HOW THE FILE IS ORGANISED
# --------------------------
# The file is split into a public API and private helpers.
# The public function load_and_clean() is the only function
# other modules need to call. Internally it calls six private
# helper functions in sequence, one per acceptance criterion.
# Each helper does one job and one job only — this keeps the
# code easy to read, test, and maintain.
#
# WHAT WE WILL DO IN THE UPCOMING SECTIONS
# -----------------------------------------
# 1. Import the tools we need (pandas, json, logging, etc.)
# 2. Set up logging so the program can record its activity.
# 3. Define constants (column names used throughout).
# 4. Implement the public load_and_clean() orchestrator.
# 5. Implement each private helper in pipeline order.
# 6. Add a CLI entry point so the file can be run directly.
"""


# ============================================================
# SECTION: Imports
# ============================================================
#
# We are now importing the external libraries this module needs.
#
# WHY THIS SECTION IS NECESSARY
# ------------------------------
# Python does not include data-processing or file-path tools
# by default. We need to explicitly import them at the top of
# the file so they are available everywhere below.
#
# WHAT EACH IMPORT DOES
# ----------------------
# - json     : Reads the category_mapping.json config file.
# - logging  : Records what the program is doing at runtime,
#              like a running diary. This is better than print()
#              for production code because it adds timestamps,
#              log levels, and can be redirected to a file.
# - os       : Provides operating-system utilities. Imported
#              for completeness; Path (below) handles most
#              file-system work in this file.
# - sys      : Gives access to command-line arguments (sys.argv)
#              and lets us exit the process with a specific code
#              (sys.exit).
# - pathlib.Path : A modern, cross-platform way to work with
#              file paths. Far safer than manually joining
#              strings with slashes.
# - pandas   : The core data-processing library. A DataFrame
#              is pandas' table-like data structure — every row
#              is a sales record, every column is a field.
#
# HOW THIS CONNECTS FORWARD
# --------------------------
# Every section below uses at least one of these imports.
# Without them, nothing in the file would work.

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
# We are now setting up the module's logging system.
#
# WHY WE NEED LOGGING
# --------------------
# When this module runs on 76 000 rows, a lot happens silently.
# Logging gives us a timestamped record of every significant
# event — files loaded, rows dropped, categories relabelled —
# without interrupting the normal flow of the program.
#
# HOW IT WORKS
# ------------
# logging.basicConfig() configures the *root* logger (the
# global default). It sets the minimum level to INFO, meaning
# DEBUG messages are ignored but INFO, WARNING, and ERROR
# messages are shown.
#
# logging.getLogger("data_cleaner") creates a named child
# logger specifically for this module. Using a named logger
# (rather than the root logger directly) means that when this
# module is imported by main.py, its log messages are clearly
# labelled "data_cleaner" in the output.
#
# The format string controls what each log line looks like:
#   2026-09-13 22:29:23 [INFO] data_cleaner: Loaded 76000 rows...
#
# WHAT COMES NEXT
# ---------------
# We define the constants that every function in this file
# will rely on.

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
# We are now defining the fixed values shared across the whole
# module.
#
# WHY WE USE CONSTANTS
# ---------------------
# Column names like "Store ID" and "Units Sold" appear in many
# functions. If we hard-code the string "Store ID" in ten
# different places and the CSV column is ever renamed, we would
# need to hunt down ten different locations. Defining it once
# as a constant means we change it in one place and every
# function automatically uses the new name.
#
# WHAT EACH CONSTANT REPRESENTS
# -------------------------------
# REQUIRED_COLUMNS : The 14 columns that every cleaned row
#   must have. If any of these are missing or null in a row,
#   that row is removed. The "Demand" column in the raw CSV
#   is intentionally excluded here — it is an extra column
#   that is not listed in the project requirements, so we do
#   not treat it as required (though we do keep it).
#
# DATE_COLUMN : The name of the date field. Stored separately
#   because it also serves as the DataFrame index after
#   cleaning, and it appears in the duplicate-key definition.
#
# DUPLICATE_KEY_COLUMNS : The three columns that together
#   uniquely identify one sales record. Two rows sharing all
#   three values are true duplicates.
#
# _DEFAULT_CONFIG_PATH : The expected location of the category-
#   mapping JSON file. We build this path dynamically using
#   __file__ (this script's own path) so it works regardless
#   of which directory the user runs the script from.
#   The leading underscore signals that this is intended for
#   internal use within the module (Python convention).
#
# WHAT COMES NEXT
# ---------------
# We implement load_and_clean(), the single public function
# that orchestrates the entire cleaning pipeline.

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

# Build the default config path relative to this file's location.
# Path(__file__).resolve() gives the absolute path to data_cleaner.py.
# .parent is the data_preprocessing/ folder.
# .parent again is the src/ folder.
# .parent once more is the project root.
# We then navigate into config/category_mapping.json.
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "category_mapping.json"


# ============================================================
# SECTION: Public API — load_and_clean()
# ============================================================
#
# We are now implementing the only function that other modules
# will ever call from this file.
#
# WHY THIS FUNCTION EXISTS
# -------------------------
# We want the outside world to have a simple, single entry
# point: "give me a path to your CSV and I will give you back
# a clean DataFrame." The caller does not need to know about
# the six individual cleaning steps — those are implementation
# details. This is the principle of encapsulation: hide
# complexity behind a clean interface.
#
# HOW IT WORKS AS AN ORCHESTRATOR
# ---------------------------------
# load_and_clean() does not do any cleaning itself. It calls
# six private helper functions in the correct order and passes
# the results of each step into the next. This is called a
# pipeline pattern: the output of one stage is the input of
# the next.
#
#   _load_csv()               -> raw DataFrame
#   _load_category_mapping()  -> mapping config (loaded early
#                                 so we fail fast if it's missing)
#   _remove_missing_values()  -> DataFrame with no null rows
#   _drop_duplicates()        -> DataFrame with no duplicate keys
#   _relabel_categories()     -> DataFrame with filling-station
#                                 category names
#   sort + set_index          -> datetime64 index, sorted asc.
#   _write_report()           -> summary printed/saved
#
# NOTE ON AC4 ORDERING:
# We call _load_category_mapping() BEFORE _remove_missing_values()
# on purpose. The requirement says "if the config file is
# absent, raise an error WITHOUT modifying any rows." Loading
# the config early means we can fail before we have touched
# the data at all — which is what the requirement demands.
#
# WHAT COMES NEXT
# ---------------
# We implement the six private helpers in the order they are
# called by load_and_clean().

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
    # --- Step 1 (AC1): Load the raw CSV ---
    # _load_csv returns both the DataFrame and the original row count.
    # We keep the original count here because later cleaning steps will
    # shrink the DataFrame, and we need the original number for the
    # final report and for the 20% missing-value threshold check.
    df, original_row_count = _load_csv(csv_path)

    # --- Step 2 (AC4 pre-check): Load the category config early ---
    # We deliberately load this BEFORE touching the data.
    # If the config file is missing, we want to raise an error
    # immediately rather than after we have already modified rows.
    category_mapping = _load_category_mapping(config_path)

    # --- Step 3 (AC2): Remove rows with missing required values ---
    # Returns the trimmed DataFrame and a per-column missing count
    # dictionary used later in the report.
    df, missing_counts = _remove_missing_values(df, original_row_count)

    # --- Step 4 (AC3): Remove duplicate rows ---
    # Returns the deduplicated DataFrame and the total removed count.
    df, duplicate_count = _drop_duplicates(df)

    # --- Step 5 (AC4 apply): Relabel category names ---
    # Returns the relabelled DataFrame, a record of every relabelling
    # decision, and a count of any rows filtered out entirely.
    df, relabel_counts, filter_count = _relabel_categories(df, category_mapping)

    # --- Step 6 (AC5): Sort by date and set as index ---
    # sort_values ensures chronological order (oldest row first).
    # set_index promotes the Date column to the DataFrame's index,
    # which is what time-series operations downstream expect.
    # The extra pd.to_datetime() call is a safety guarantee —
    # in rare edge cases pandas can infer a different dtype after
    # set_index, so we explicitly re-assert datetime64.
    df = df.sort_values(DATE_COLUMN).set_index(DATE_COLUMN)
    df.index = pd.to_datetime(df.index)  # guarantee datetime64 index

    # --- Step 7 (AC6): Produce the summary report ---
    # We capture the final row count here (after all cleaning) so the
    # report can show: original count, cleaned count, total removed.
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

    # Return the trusted, cleaned DataFrame to the caller.
    return df


# ============================================================
# SECTION: Private Helpers
# ============================================================
#
# We are now implementing the six private helper functions that
# load_and_clean() delegates its work to.
#
# WHY PRIVATE HELPERS?
# ---------------------
# Breaking a complex task into small, single-purpose functions
# has three benefits:
#   1. Readability  — each function's name tells you exactly
#      what it does.
#   2. Testability  — you can test _drop_duplicates() in
#      isolation without running the entire pipeline.
#   3. Maintainability — if the duplicate-removal logic needs
#      to change, you only edit one small function.
#
# The leading underscore in each name (_load_csv, etc.) is a
# Python convention that signals: "this is an internal detail,
# not part of the public API." Other modules should not call
# these directly.
#
# We implement them in pipeline order for readability.


# ============================================================
# SECTION: Private Helper — _load_csv  (AC1)
# ============================================================
#
# We are now implementing the first step: loading the file.
#
# WHY THIS IS THE FIRST STEP
# ---------------------------
# Everything depends on successfully reading the raw data into
# memory. If the file does not exist or is corrupted, there is
# no point continuing. We fail fast and give a clear error
# message that tells the user exactly what went wrong.
#
# HOW IT WORKS
# ------------
# 1. Convert the path string to a Path object for safe,
#    OS-independent file operations.
# 2. Check the file exists — raise FileNotFoundError if not.
# 3. Use pd.read_csv() with parse_dates to tell pandas to
#    convert the "Date" column from a plain string like
#    "2022-01-01" into a Python/pandas datetime object.
# 4. Double-check that the Date column really did become a
#    datetime type — if the column had mixed formats pandas
#    might silently leave it as a string.
# 5. Return the DataFrame AND the row count so the caller
#    always has the original size for comparison.
#
# WHAT COMES NEXT
# ---------------
# _load_category_mapping() loads the config file that controls
# how categories are renamed.

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

    # If the file does not exist, raise immediately with a message
    # that includes the bad path so the user knows exactly what to fix.
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: '{csv_path}'. "
            "Please verify the path and try again."
        )

    # Wrap pd.read_csv in a try/except so that any low-level parsing
    # error (wrong encoding, binary file, etc.) is caught and re-raised
    # as a friendlier ValueError with context.
    try:
        df = pd.read_csv(path, parse_dates=[DATE_COLUMN])
    except Exception as exc:
        raise ValueError(
            f"Cannot parse '{csv_path}' as a CSV file. Reason: {exc}"
        ) from exc

    # Verify that pandas actually converted the Date column to datetime.
    # parse_dates is best-effort — if the dates were in an unusual format,
    # pandas may have left them as strings without raising an error.
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
# We are now loading the configuration that tells us how to
# rename the dataset's generic retail categories into names
# that make sense for a filling-station forecourt.
#
# WHY THIS IS A SEPARATE STEP
# ----------------------------
# The category-renaming rules live in an external JSON file
# (config/category_mapping.json) rather than being hard-coded
# here. That design decision means a project manager can change
# the mappings without touching the Python source code.
#
# We load this config early (before any data modification)
# because the requirement states: "if the config file is
# absent, raise an error WITHOUT modifying any rows." If we
# loaded it later — after removing missing values and
# duplicates — we would have already changed the data before
# discovering the config was missing. Loading it here ensures
# we either have everything we need or we fail cleanly.
#
# HOW IT WORKS
# ------------
# 1. Check the config file exists — raise ValueError if not.
#    (We raise ValueError rather than FileNotFoundError here
#    because from the caller's perspective this is a
#    configuration error, not a missing data file.)
# 2. Open and parse the JSON file.
# 3. Validate that the two required keys ("mappings" and
#    "valid_target_categories") are present.
# 4. Return the whole config dict so _relabel_categories()
#    can use both keys later.
#
# WHAT COMES NEXT
# ---------------
# _remove_missing_values() scans every required column for
# null values and removes the affected rows.

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

    # Config existence check — fail before touching any data.
    if not path.exists():
        raise ValueError(
            f"Category-mapping configuration not found: '{config_path}'. "
            "Cannot proceed without it."
        )

    # Parse the JSON. If the file is malformed (e.g. a syntax error
    # in the JSON), json.load raises an exception. We catch it and
    # re-raise as a descriptive ValueError.
    try:
        with open(path, encoding="utf-8") as fh:
            config = json.load(fh)
    except Exception as exc:
        raise ValueError(
            f"Cannot read category-mapping config '{config_path}'. Reason: {exc}"
        ) from exc

    # Validate that both expected keys exist in the config.
    # A config with only one key would cause a KeyError later in
    # _relabel_categories() — better to catch it here with a clear message.
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
# We are now scanning the DataFrame for missing (null) values
# in any required column and removing the affected rows.
#
# WHY MISSING VALUES ARE A PROBLEM
# ----------------------------------
# Machine learning models and statistical calculations cannot
# work with null values. A null price, for example, would
# cause a regression model to either crash or silently produce
# incorrect predictions. We remove these rows early so that
# every downstream function receives guaranteed-complete data.
#
# WHY THERE IS A 20% THRESHOLD
# -----------------------------
# Removing a handful of rows is normal and acceptable. But if
# 30% or 40% of the dataset has missing values, something is
# fundamentally wrong — maybe the wrong file was passed, or
# the CSV was exported incorrectly. Quietly removing 30 000
# rows from a 76 000-row dataset would bias every model we
# train. The 20% threshold is a safeguard that forces us to
# investigate before proceeding.
#
# HOW IT WORKS
# ------------
# 1. Build a list of required columns that actually exist in
#    the DataFrame (defensive check — the CSV might not have
#    every column we expect).
# 2. Loop over each column and count null values individually
#    — this gives us the per-column detail for the report.
# 3. Calculate what percentage of ALL rows have at least one
#    null in any required column.
# 4. Raise ValueError if that percentage exceeds 20%.
# 5. Drop all rows that have any null in any required column
#    using df.dropna(subset=...).
# 6. Return the cleaned DataFrame and the per-column counts.
#
# KEY CONCEPT — axis=1 in .any(axis=1)
# -------------------------------------
# df[cols].isna() produces a table of True/False values.
# .any(axis=1) collapses each ROW into a single True/False:
# True if ANY column in that row was null. We then .sum() to
# count how many rows had at least one null.
#
# WHAT COMES NEXT
# ---------------
# _drop_duplicates() removes rows that share the same
# (Date, Store ID, Product ID) combination.

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
    # Build a list of only the required columns that are actually present.
    # This prevents a KeyError if the CSV happens to be missing a column header.
    cols_to_check = [c for c in REQUIRED_COLUMNS if c in df.columns]
    missing_counts = {}

    # Count nulls per column for the report. We do this before dropping
    # so we capture the original counts, not the post-drop counts.
    for col in cols_to_check:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            missing_counts[col] = int(n_missing)
            logger.info("Column '%s': %d missing value(s) found.", col, n_missing)

    # Count how many rows have at least one null across all required columns.
    # axis=1 means "look across columns for each row".
    rows_with_any_missing = df[cols_to_check].isna().any(axis=1).sum()

    # Express as a fraction of the ORIGINAL row count (not the current size)
    # so the threshold is always measured against the raw dataset.
    missing_pct = rows_with_any_missing / original_row_count if original_row_count > 0 else 0

    # AC2 safeguard: if more than 20% of rows are affected, abort.
    if missing_pct > 0.20:
        raise ValueError(
            f"Missing-value rate is {missing_pct:.1%} ({rows_with_any_missing} rows), "
            "which exceeds the 20% threshold. Cannot produce a cleaned DataFrame."
        )

    # Drop every row that has a null in at least one required column.
    # subset= tells dropna to only consider the required columns —
    # nulls in optional/extra columns (like "Demand") are left alone.
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
# We are now removing rows that represent the same sales event
# recorded more than once.
#
# WHAT IS A DUPLICATE HERE?
# --------------------------
# A duplicate is defined as two or more rows that share the
# exact same values for Date, Store ID, AND Product ID. That
# combination should be unique — you cannot sell the same
# product at the same store on the same day twice as separate
# independent events. If you find two such rows, one of them
# is a data-entry error or a re-export artifact.
#
# WHY WE KEEP THE FIRST OCCURRENCE
# ----------------------------------
# Without additional context we cannot know which of the
# duplicates is "correct". Keeping the first occurrence is a
# consistent, reproducible rule. If the data has already been
# sorted chronologically, "first" also means "earliest import",
# which is often the most reliable record.
#
# HOW IT WORKS
# ------------
# drop_duplicates(subset=..., keep="first") is pandas' built-in
# deduplication. subset= defines which columns to look at when
# deciding if two rows are identical. keep="first" means the
# first row of each group is retained and all subsequent
# copies are dropped.
#
# We capture the row count before and after to know how many
# were removed — this number goes into the report.
#
# WHAT COMES NEXT
# ---------------
# _relabel_categories() translates generic retail category
# names into filling-station equivalents.

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

    # Drop rows where all three key columns are identical to a prior row.
    # keep="first" preserves the earliest occurrence of each group.
    df = df.drop_duplicates(subset=DUPLICATE_KEY_COLUMNS, keep="first")

    after = len(df)
    duplicate_count = before - after  # how many rows were silently removed

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
# We are now translating the Kaggle dataset's generic retail
# category names into labels that fit a filling-station context.
#
# WHY WE NEED TO RELABEL
# -----------------------
# The dataset was built for a general retail store. Its
# categories ("Electronics", "Furniture", etc.) mean nothing
# to a filling-station manager who stocks Snacks, Beverages,
# Airtime cards, and Motor Accessories. Relabelling makes the
# system's outputs meaningful to the actual user.
#
# WHY THIS IS DATA-DRIVEN (config file) NOT HARD-CODED
# -----------------------------------------------------
# The mapping lives in category_mapping.json, not inside
# this function. That means a non-programmer can update the
# mappings without touching code. It also means the unit
# tests can swap in a different config to test edge cases.
#
# HOW IT WORKS — TWO PHASES
# --------------------------
# Phase 1 — Relabelling:
#   We loop over every (original_name -> new_name) pair in
#   the config's "mappings" dict. For each pair we build a
#   boolean mask (True wherever Category == original_name)
#   and use df.loc[mask, "Category"] = new_name to overwrite
#   those cells. We record how many rows were changed.
#
# Phase 2 — Filtering:
#   After relabelling, any row whose Category is NOT in
#   valid_target_categories is removed entirely. This handles
#   the case where the dataset contains a category that was
#   not in our mapping at all — we do not want mystery
#   categories leaking into the cleaned output.
#
# KEY CONCEPT — boolean mask
# ---------------------------
# mask = df["Category"] == "Groceries"
# This creates a Series of True/False values, one per row.
# df.loc[mask, "Category"] = "Snacks" then writes "Snacks"
# only to the rows where mask is True. It is the pandas
# equivalent of: "for every row where Category is Groceries,
# change it to Snacks."
#
# WHAT COMES NEXT
# ---------------
# Back in load_and_clean(), we sort the DataFrame by date
# and promote the Date column to the index. Then _write_report()
# produces the final summary.

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

    # Dictionary to record every relabelling decision for the report.
    # Structure: { "Groceries": ("Snacks", 30400), ... }
    relabel_counts = {}

    # --- Phase 1: Relabelling ---
    for original_cat, new_cat in mappings.items():
        # Boolean mask: True for every row whose Category matches original_cat.
        mask = df["Category"] == original_cat
        count = int(mask.sum())  # convert numpy int to plain Python int for JSON safety

        if count > 0:
            # Overwrite Category values in-place for the matching rows.
            df.loc[mask, "Category"] = new_cat
            relabel_counts[original_cat] = (new_cat, count)
            logger.info(
                "Relabelled %d row(s): '%s' -> '%s'.", count, original_cat, new_cat
            )

    # --- Phase 2: Filtering ---
    # Remove any remaining rows whose Category did not appear in the
    # mappings (so they were never relabelled) and is therefore not
    # a valid target category for the filling-station context.
    before = len(df)
    df = df[df["Category"].isin(valid_targets)]  # keep only valid-target rows
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
# We are now producing the human-readable summary that tells
# anyone reading the output exactly what the cleaner did.
#
# WHY A REPORT MATTERS
# ---------------------
# Data cleaning is invisible — the cleaned DataFrame looks
# similar to the raw one. Without a report, a data scientist
# reviewing the output would have no idea that 12 160 rows
# were relabelled or that 0 duplicates were found. The report
# creates a written audit trail of every decision made.
#
# THE 10 000-CHARACTER LIMIT
# ---------------------------
# The requirement caps the report at 10 000 characters. This
# prevents the report from becoming unreadably large for
# datasets with hundreds of columns. We enforce it by slicing
# the string and appending a truncation notice.
#
# HOW IT WORKS
# ------------
# 1. Build the report as a list of strings (lines), then join
#    them with newlines at the end. Building a list first is
#    more efficient than repeatedly concatenating strings
#    (each string concatenation in Python creates a new object
#    in memory).
# 2. Conditionally add detail lines — if missing_counts is
#    empty, print "No missing values detected" instead of
#    leaving a blank section.
# 3. Truncate if the report exceeds 10 000 characters.
# 4. Always print to stdout (the terminal).
# 5. Optionally write to a file if report_path was provided.
#    We create the parent directory automatically so the
#    caller does not need to pre-create the outputs/ folder.
#
# WHAT COMES NEXT
# ---------------
# After _write_report() returns, load_and_clean() returns the
# cleaned DataFrame to the caller. The next section implements
# the CLI entry point.

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
    # Build the report line by line. Using a list and joining once at
    # the end is more memory-efficient than repeated string concatenation.
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

    # Conditionally expand the missing-value section.
    # If no columns had missing values, we say so explicitly
    # rather than leaving the section empty and confusing.
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

    # Conditionally expand the relabelling section.
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

    # Join all lines into one string for printing and writing.
    report = "\n".join(lines)

    # Enforce the 10 000-character cap required by AC6.
    # We truncate at 9 990 (not 10 000) to leave room for the notice itself.
    if len(report) > 10_000:
        report = report[:9_990] + "\n[... report truncated to 10 000 characters ...]"

    # Always print to stdout so the user sees the report in their terminal.
    print(report)

    # If a file path was provided, write the report there too.
    if report_path:
        report_file = Path(report_path)
        # mkdir(parents=True, exist_ok=True) creates the full directory
        # chain (e.g. outputs/) without failing if it already exists.
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as fh:
            fh.write(report)
        logger.info("Summary report saved to '%s'.", report_path)


# ============================================================
# SECTION: CLI Entry Point
# ============================================================
#
# We are now adding the code that runs when a person executes
# this file directly from the terminal:
#
#     python src/data_preprocessing/data_cleaner.py data/raw/sales_data.csv
#
# WHY THIS PATTERN EXISTS
# ------------------------
# In Python, every file can be both imported as a module AND
# run directly as a script. The if __name__ == "__main__":
# guard ensures that the CLI code only executes when the file
# is run directly. When another module does
# "from data_cleaner import load_and_clean", the CLI block
# is skipped entirely — only the functions are loaded.
#
# Without this guard, simply importing data_cleaner would
# immediately try to read sys.argv and run the pipeline,
# which would crash any importing module.
#
# HOW THE ARGUMENT HANDLING WORKS
# --------------------------------
# sys.argv is a list where:
#   sys.argv[0] = the script name ("data_cleaner.py")
#   sys.argv[1] = first argument (csv_path, required)
#   sys.argv[2] = second argument (config_path, optional)
#   sys.argv[3] = third argument (report_path, optional)
#
# We slice off the script name with sys.argv[1:] to get just
# the user-supplied arguments. If none were supplied, we print
# a usage hint to stderr (not stdout — usage errors belong on
# stderr) and exit with code 1 (the convention for "error").
#
# For optional arguments we use a conditional expression:
#   args[1] if len(args) > 1 else default_value
# This is Python's equivalent of a ternary operator.
#
# HOW ERRORS ARE HANDLED
# -----------------------
# We catch only FileNotFoundError and ValueError — the two
# error types that load_and_clean() can raise. If something
# unexpected happens (e.g. an out-of-memory error), we let it
# propagate naturally so the full traceback is visible.
# Catching Exception silently swallowing unexpected bugs is
# a common anti-pattern we deliberately avoid here.
#
# WHAT COMES NEXT
# ---------------
# This is the end of the file. See the End-of-File Summary
# below for a recap of what we built and what comes next
# in the broader project.

if __name__ == "__main__":
    # Slice off sys.argv[0] (the script filename) — we only want user args.
    args = sys.argv[1:]

    # If the user ran the script with no arguments, show usage and quit.
    if not args:
        print(
            "Usage: python src/data_preprocessing/data_cleaner.py data/raw/sales_data.csv [config_path] [report_path]",
            file=sys.stderr,  # usage errors go to stderr, not stdout
        )
        sys.exit(1)  # exit code 1 = something went wrong

    # Assign arguments, falling back to defaults for optional ones.
    _csv = args[0]
    _config = args[1] if len(args) > 1 else str(_DEFAULT_CONFIG_PATH)
    _report = (
        args[2]
        if len(args) > 2
        else str(
            # Default report location: outputs/cleaning_report.txt
            # relative to the project root (three levels up from data_preprocessing/).
            Path(__file__).resolve().parent.parent.parent / "outputs" / "cleaning_report.txt"
        )
    )

    try:
        cleaned_df = load_and_clean(_csv, _config, _report)

        # Print a brief confirmation so the user knows it worked.
        print(f"\nCleaning complete. DataFrame shape: {cleaned_df.shape}")
        print(f"Date index range: {cleaned_df.index.min()} to {cleaned_df.index.max()}")
        print(f"Categories present: {sorted(cleaned_df['Category'].unique())}")

    except (FileNotFoundError, ValueError) as err:
        # Log the error at ERROR level and exit with code 1.
        # We do NOT print a traceback here — a clean error message is
        # more user-friendly than a wall of Python internals.
        logger.error("Cleaning failed: %s", err)
        sys.exit(1)


# ============================================================
# END-OF-FILE SUMMARY
# ============================================================
#
# WHAT WE ACCOMPLISHED IN THIS FILE
# -----------------------------------
# We built data_cleaner.py — the first and most foundational
# module of the AI Retail Predictors system. This module is
# responsible for turning a raw, general-purpose Kaggle CSV
# into a clean, trusted pandas DataFrame that every other
# module in the pipeline can rely on.
#
# THE MAJOR SECTIONS WE IMPLEMENTED
# -----------------------------------
# 1. Imports
#    Brought in json, logging, sys, pathlib, and pandas —
#    the tools the module needs to do its work.
#
# 2. Logger Setup
#    Configured a named logger ("data_cleaner") so all
#    runtime activity is timestamped and clearly attributed.
#
# 3. Constants
#    Defined REQUIRED_COLUMNS, DATE_COLUMN, and
#    DUPLICATE_KEY_COLUMNS once, centrally, so every function
#    uses the same source of truth.
#
# 4. Public API — load_and_clean()
#    The single entry point for callers. Orchestrates the
#    six cleaning steps in the correct order and returns the
#    cleaned DataFrame.
#
# 5. _load_csv()        [AC1]
#    Loads the CSV, parses dates, and raises clear errors if
#    the file is missing or unreadable.
#
# 6. _load_category_mapping()  [AC4 pre-check]
#    Loads the JSON config early so we fail before modifying
#    any data if the config is absent.
#
# 7. _remove_missing_values()  [AC2]
#    Removes null rows, logs per-column counts, and enforces
#    the 20% data-quality threshold.
#
# 8. _drop_duplicates()  [AC3]
#    Removes repeated (Date, Store ID, Product ID) rows,
#    keeping the first occurrence.
#
# 9. _relabel_categories()  [AC4 apply]
#    Translates generic retail categories to filling-station
#    equivalents and filters unmapped rows.
#
# 10. _write_report()  [AC6]
#     Builds and prints the cleaning audit trail; optionally
#     saves it to outputs/cleaning_report.txt.
#
# 11. CLI Entry Point
#     Allows the module to be run directly from the terminal
#     with a file path argument.
#
# HOW THE SECTIONS WORK TOGETHER
# --------------------------------
# load_and_clean() acts as the pipeline manager. Each private
# helper does exactly one job, passes its output to the next
# step via load_and_clean(), and records what it did for the
# final report. The data flows like this:
#
#   Raw CSV
#     -> _load_csv()               : DataFrame + original count
#     -> _load_category_mapping()  : config dict (fail-fast)
#     -> _remove_missing_values()  : trimmed DataFrame
#     -> _drop_duplicates()        : deduplicated DataFrame
#     -> _relabel_categories()     : filling-station categories
#     -> sort + set_index          : datetime64 index, asc order
#     -> _write_report()           : printed + saved audit trail
#     -> return cleaned DataFrame
#
# WHAT HAS BEEN ACHIEVED SO FAR
# -------------------------------
# The system can now ingest the raw Kaggle sales_data.csv and
# produce a single clean, consistently labelled DataFrame with:
#   - 76 000 rows (no rows lost from this particular dataset)
#   - A datetime64 index sorted oldest to newest
#   - 4 filling-station categories: Airtime, Beverages,
#     Motor Accessories, Snacks
#   - No missing values in any of the 14 required columns
#   - No duplicate (Date, Store ID, Product ID) records
#
# NEXT STEPS
# ----------
# 1. Requirement 2 — Chronological Train/Test Split:
#    Use the cleaned DataFrame produced here to split the data
#    80/20 by date, with no future data leaking into the
#    training set.
#
# 2. Requirement 3 — Time-Series Decomposition:
#    Apply seasonal_decompose from statsmodels to the training
#    set to extract trend, seasonality, and residual components.
#
# 3. Requirements 4 & 5 — ML and DL Model Training:
#    Train a Random Forest / Gradient Boosting model and an
#    LSTM/Dense neural network on the training split.
#
# The next logical file to implement is:
#   src/time_series_analyser.py  (after the train/test split)
# ============================================================
