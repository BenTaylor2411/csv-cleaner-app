import pandas as pd
from datetime import datetime


def slugify_cols(cols):
    out = []
    for c in cols:
        c2 = str(c).strip().lower().replace(" ", "_").replace("-", "_")
        while "__" in c2:
            c2 = c2.replace("__", "_")
        out.append(c2)
    return out


def trim_strings(df):
    cols = df.select_dtypes(include=["object", "string"]).columns
    for col in cols:
        df[col] = df[col].astype("string").str.strip()
    return list(cols)


def attempt_numeric_coerce(df):
    converted = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].astype("string").str.replace(",", "", regex=False)
        n = pd.to_numeric(s, errors="coerce")
        if n.notna().mean() >= 0.85 and n.notna().sum() > 0:
            df[col] = n
            converted.append(col)
    return converted


def fill_missing(df):
    filled = {"numeric": [], "string": []}

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and s.isna().any():
            df[col] = s.fillna(s.median())
            filled["numeric"].append(col)
        elif s.isna().any():
            df[col] = s.astype("string").fillna("Unknown")
            filled["string"].append(col)

    return filled


def analyze_csv(path):
    df = pd.read_csv(path)
    df.columns = slugify_cols(df.columns)

    trimmed = trim_strings(df)
    numeric = attempt_numeric_coerce(df)
    filled = fill_missing(df)

    report = []
    report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    report.append(f"Rows: {len(df)}")
    report.append(f"Columns: {len(df.columns)}")
    report.append("")
    report.append(f"Trimmed columns: {trimmed}")
    report.append(f"Numeric conversions: {numeric}")
    report.append(f"Filled missing values: {filled}")

    return df, "\n".join(report)


if __name__ == "__main__":
    print("CSV cleaner core logic ready.")
