import pandas as pd

PATH = "get_data/output/btcusdt_1m_spot_clean.parquet"

# --- load ---
if PATH.endswith(".parquet"):
    df = pd.read_parquet(PATH)
elif PATH.endswith(".csv"):
    df = pd.read_csv(PATH, parse_dates=["ts"])
else:
    raise ValueError("Unsupported file format")

# --- ensure datetime + UTC ---
df["ts"] = pd.to_datetime(df["ts"], utc=True)

# --- basic info ---
print("=== COLUMNS ===")
print(df.columns.tolist())

print("\n=== DTYPES ===")
print(df.dtypes)

print("\n=== SHAPE ===")
print(df.shape)

# --- head / tail ---
print("\n=== FIRST 20 ROWS ===")
print(df.head(20))

print("\n=== LAST 20 ROWS ===")
print(df.tail(20))

# --- optional: quick sanity checks ---
print("\n=== TIME RANGE ===")
print("from:", df["ts"].iloc[0])
print("to  :", df["ts"].iloc[-1])

print("\n=== DUPLICATE TS ===")
print(df["ts"].duplicated().sum())

print("\n=== TIME STEP (mode) ===")
print(df["ts"].diff().mode())
