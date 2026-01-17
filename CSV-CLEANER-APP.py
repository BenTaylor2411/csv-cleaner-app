import os
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


def clean_and_export(csv_path, outdir):
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = slugify_cols(df.columns)

    trimmed = trim_strings(df)
    numeric = attempt_numeric_coerce(df)
    filled = fill_missing(df)

    base = os.path.splitext(os.path.basename(csv_path))[0]
    cleaned_path = os.path.join(outdir, f"{base}_cleaned.csv")
    summary_path = os.path.join(outdir, f"{base}_summary.txt")

    df.to_csv(cleaned_path, index=False)

    report = []
    report.append("CSV CLEAN REPORT")
    report.append("=" * 20)
    report.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    report.append("")
    report.append(f"Rows: {len(df)}")
    report.append(f"Columns: {len(df.columns)}")
    report.append("")
    report.append(f"Trimmed columns: {trimmed}")
    report.append(f"Numeric conversions: {numeric}")
    report.append(f"Filled missing values: {filled}")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    return cleaned_path, summary_path


if __name__ == "__main__":
    print("CSV cleaner now exports cleaned data and reports.")
