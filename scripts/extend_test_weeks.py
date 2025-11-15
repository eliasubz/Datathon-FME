# %% START
import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INPUT_PATH = DATA_DIR / "test.csv"
OUTPUT_PATH = DATA_DIR / "test_extended.csv"

DATE_FORMAT = "%d/%m/%Y"


def iso_week_range(start, end):
    """Yield (year, iso_week) pairs for all ISO weeks from start to end inclusive.

    Weeks are represented by the Monday of each ISO week.
    """
    # Normalize to Monday of ISO week
    start_monday = start - pd.to_timedelta(start.dayofweek, unit="D")
    end_monday = end - pd.to_timedelta(end.dayofweek, unit="D")

    current = start_monday
    while current <= end_monday:
        iso = current.isocalendar()
        yield int(iso.year), int(iso.week)
        current += pd.to_timedelta(7, unit="D")

# %% MAIN SCRIPT

print("Extending test weeks...")
df = pd.read_csv(INPUT_PATH, delimiter=";")

# Parse dates
phase_in = pd.to_datetime(df["phase_in"], format=DATE_FORMAT, errors="coerce")
phase_out = pd.to_datetime(df["phase_out"], format=DATE_FORMAT, errors="coerce")

# Replace invalid dates with themselves to avoid dropping entire rows later
df = df.copy()
df["phase_in_dt"] = phase_in
df["phase_out_dt"] = phase_out

# Build extended rows
extended_rows = []

for _, row in df.iterrows():
    start = row["phase_in_dt"]
    end = row["phase_out_dt"]
    if pd.isna(start) or pd.isna(end):
        continue

    for year, week in iso_week_range(start, end):
        new_row = row.copy()
        new_row["year"] = year
        new_row["num_week_iso"] = week
        extended_rows.append(new_row)

if not extended_rows:
    # Nothing to write
    print("No rows generated; check phase_in/phase_out values in test.csv")

extended_df = pd.DataFrame(extended_rows)

# Ensure column order: start from original df columns, then append year/num_week_iso,
# and finally drop helper columns used only for date handling.
cols = [c for c in df.columns if c not in {"year", "num_week_iso"}]
cols += ["year", "num_week_iso"]
extended_df = extended_df[cols]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
extended_df.to_csv(OUTPUT_PATH, sep=";", index=False)
print(f"Extended test data written to {OUTPUT_PATH} with {len(extended_df)} rows")

extended_df.head()

