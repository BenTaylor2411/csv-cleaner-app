import os
import sys
import threading
import subprocess
from datetime import datetime

import pandas as pd

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def slugify_cols(cols):
    out = []
    for c in cols:
        c2 = str(c).strip().lower()
        c2 = c2.replace(" ", "_").replace("-", "_")
        while "__" in c2:
            c2 = c2.replace("__", "_")
        out.append(c2)
    return out


def trim_strings(df):
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].astype("string").str.strip()
    return list(str_cols)


def attempt_numeric_coerce(df, max_cols=25):
    converted = []
    cand = list(df.select_dtypes(include=["object", "string"]).columns)[:max_cols]
    for col in cand:
        low = col.lower()
        if any(k in low for k in ["id", "uuid", "postcode", "zip"]):
            continue

        series = df[col].astype("string").str.replace(",", "", regex=False)
        as_num = pd.to_numeric(series, errors="coerce")
        ratio = as_num.notna().mean()

        if ratio >= 0.85 and as_num.notna().sum() > 0:
            df[col] = as_num
            converted.append(col)
    return converted


def try_parse_dates(df, max_cols=12):
    date_like = []
    for col in df.columns:
        name = col.lower()
        if any(k in name for k in ["date", "time", "timestamp", "created", "updated", "login"]):
            date_like.append(col)

    date_like = date_like[:max_cols]
    parsed = []

    for col in date_like:
        try:
            before_na = df[col].isna().sum()
            dt = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            after_na = dt.isna().sum()

            if after_na > before_na + int(len(df) * 0.25):
                df[col] = df[col].astype("string")
            else:
                df[col] = dt
                parsed.append(col)
        except Exception:
            df[col] = df[col].astype("string")

    return parsed


def fill_missing(df):
    report = {"numeric_filled": [], "string_filled": [], "bool_filled": [], "left_blank": []}

    for col in df.columns:
        s = df[col]

        if pd.api.types.is_numeric_dtype(s):
            if s.isna().any():
                med = s.median(skipna=True)
                if pd.isna(med):
                    report["left_blank"].append(col)
                else:
                    df[col] = s.fillna(med)
                    report["numeric_filled"].append(col)

        elif pd.api.types.is_bool_dtype(s):
            if s.isna().any():
                df[col] = s.fillna(False)
                report["bool_filled"].append(col)

        elif pd.api.types.is_datetime64_any_dtype(s):
            if s.isna().any():
                report["left_blank"].append(col)

        else:
            if s.isna().any():
                df[col] = s.astype("string").fillna("Unknown")
                report["string_filled"].append(col)

    return report


def basic_stats_text(df):
    lines = []
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns: {len(df.columns):,}")
    lines.append("")

    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    lines.append("Missing values (only columns with missing):")
    if len(missing) == 0:
        lines.append("  None 🎉")
    else:
        for col, cnt in missing.items():
            lines.append(f"  - {col}: {cnt:,}")
    lines.append("")

    lines.append("Column types:")
    for col in df.columns:
        lines.append(f"  - {col}: {df[col].dtype}")
    lines.append("")

    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) > 0:
        lines.append("Numeric summary (mean / min / max):")
        for col in num_cols:
            s = df[col]
            try:
                lines.append(f"  - {col}: mean={s.mean():.4g}, min={s.min():.4g}, max={s.max():.4g}")
            except Exception:
                lines.append(f"  - {col}: (couldn't compute)")
        lines.append("")

    return "\n".join(lines)


def clean_and_analyze(input_csv, outdir, sep=",", encoding=None):
    os.makedirs(outdir, exist_ok=True)

    try:
        df = pd.read_csv(input_csv, sep=sep, encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(input_csv, sep=sep, encoding="latin-1")

    original_rows = len(df)
    original_cols = list(df.columns)

    df.columns = slugify_cols(df.columns)

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)

    trimmed_cols = trim_strings(df)
    numeric_converted = attempt_numeric_coerce(df)
    parsed_dates = try_parse_dates(df)

    before_dupes = len(df)
    df = df.drop_duplicates()
    dupes_removed = before_dupes - len(df)

    fill_report = fill_missing(df)

    base = os.path.splitext(os.path.basename(input_csv))[0]
    cleaned_path = os.path.join(outdir, f"{base}_cleaned.csv")
    summary_path = os.path.join(outdir, f"{base}_summary.txt")
    chart_path = os.path.join(outdir, f"{base}_charts.png")

    df.to_csv(cleaned_path, index=False)

    report_lines = []
    report_lines.append("CSV CLEAN + ANALYSIS REPORT")
    report_lines.append("=" * 28)
    report_lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    report_lines.append(f"Input file: {input_csv}")
    report_lines.append(f"Output folder: {outdir}")
    report_lines.append("")
    report_lines.append("CHANGES MADE")
    report_lines.append("-" * 12)
    report_lines.append(f"Original rows: {original_rows:,}")
    report_lines.append(f"Final rows:    {len(df):,}")
    report_lines.append(f"Duplicates removed: {dupes_removed:,}")
    report_lines.append(f"Empty columns removed: {len(empty_cols)}")
    if empty_cols:
        report_lines.append(f"  -> {empty_cols}")
    report_lines.append("")
    report_lines.append("Column name standardization:")
    report_lines.append(f"  Original columns: {original_cols}")
    report_lines.append(f"  New columns:      {list(df.columns)}")
    report_lines.append("")
    report_lines.append("Type fixes:")
    report_lines.append(f"  Trimmed text columns: {trimmed_cols[:20]}{' ...' if len(trimmed_cols)>20 else ''}")
    report_lines.append(f"  Numeric conversions:  {numeric_converted}")
    report_lines.append(f"  Date columns parsed:  {parsed_dates}")
    report_lines.append("")
    report_lines.append("Missing value handling:")
    report_lines.append(f"  Numeric filled (median): {fill_report['numeric_filled']}")
    report_lines.append(f"  String filled ('Unknown'): {fill_report['string_filled']}")
    report_lines.append(f"  Bool filled (False): {fill_report['bool_filled']}")
    report_lines.append(f"  Left blank (NaT/unknown): {fill_report['left_blank']}")
    report_lines.append("")
    report_lines.append("ANALYSIS")
    report_lines.append("-" * 8)
    report_lines.append(basic_stats_text(df))
    report_text = "\n".join(report_lines)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0].head(20)

    numeric_cols = list(df.select_dtypes(include=["number"]).columns)[:3]

    return {
        "df": df,
        "report_text": report_text,
        "cleaned_path": cleaned_path,
        "summary_path": summary_path,
        "chart_path": chart_path,
        "missing_series": missing,
        "numeric_cols": numeric_cols,
        "base_name": base,
        "outdir": outdir,
    }


def open_path(path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception:
        messagebox.showinfo("Open", f"Couldn't open automatically:\n\n{path}")


class CSVCleanerRounded(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("CSV Cleaner Pro")
        self.geometry("1220x780")
        self.minsize(1080, 680)

        self.input_path = tk.StringVar(value="")
        self.outdir_path = tk.StringVar(value=os.path.join(os.getcwd(), "output"))
        self.sep_val = tk.StringVar(value=",")
        self.encoding_val = tk.StringVar(value="")

        self.last_outputs = {"cleaned": None, "summary": None, "chart": None, "outdir": None}

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()
        self._build_left_panel()
        self._build_right_panel()

        self.figure = None
        self.canvas = None

    def _build_header(self):
        header = ctk.CTkFrame(self, corner_radius=18)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=(16, 10))
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(header, text="CSV Cleaner Pro", font=ctk.CTkFont(size=20, weight="bold"))
        title.grid(row=0, column=0, sticky="w", padx=18, pady=(14, 0))

        sub = ctk.CTkLabel(header, text="One-stop cleaning, validation, preview + visuals", text_color="#A7B0C0")
        sub.grid(row=1, column=0, sticky="w", padx=18, pady=(2, 14))

        self.theme_var = tk.StringVar(value="dark")
        theme_btn = ctk.CTkSegmentedButton(
            header, values=["dark", "light"], variable=self.theme_var, command=self._toggle_theme
        )
        theme_btn.grid(row=0, column=1, rowspan=2, sticky="e", padx=18, pady=18)

    def _toggle_theme(self, _val=None):
        mode = self.theme_var.get()
        ctk.set_appearance_mode(mode)

    def _build_left_panel(self):
        left = ctk.CTkFrame(self, corner_radius=18)
        left.grid(row=1, column=0, sticky="nsw", padx=(16, 10), pady=(0, 16))
        left.grid_rowconfigure(99, weight=1)

        card1 = ctk.CTkFrame(left, corner_radius=16)
        card1.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 12))
        card1.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(card1, text="Input", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=14, pady=(12, 6)
        )

        ctk.CTkLabel(card1, text="CSV file", text_color="#A7B0C0").grid(row=1, column=0, sticky="w", padx=14)
        row_file = ctk.CTkFrame(card1, corner_radius=14, fg_color="transparent")
        row_file.grid(row=2, column=0, sticky="ew", padx=14, pady=(8, 10))
        row_file.grid_columnconfigure(0, weight=1)

        self.csv_entry = ctk.CTkEntry(row_file, textvariable=self.input_path)
        self.csv_entry.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(row_file, text="Browse", width=90, corner_radius=14, command=self.browse_csv).grid(
            row=0, column=1, padx=(10, 0)
        )

        ctk.CTkLabel(card1, text="Output folder", text_color="#A7B0C0").grid(row=3, column=0, sticky="w", padx=14)
        row_out = ctk.CTkFrame(card1, corner_radius=14, fg_color="transparent")
        row_out.grid(row=4, column=0, sticky="ew", padx=14, pady=(8, 12))
        row_out.grid_columnconfigure(0, weight=1)

        self.out_entry = ctk.CTkEntry(row_out, textvariable=self.outdir_path)
        self.out_entry.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(row_out, text="Choose", width=90, corner_radius=14, command=self.browse_outdir).grid(
            row=0, column=1, padx=(10, 0)
        )

        card2 = ctk.CTkFrame(left, corner_radius=16)
        card2.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 12))
        card2.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(card2, text="Options", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(12, 6)
        )

        ctk.CTkLabel(card2, text="Separator", text_color="#A7B0C0").grid(row=1, column=0, sticky="w", padx=14)
        self.sep_entry = ctk.CTkEntry(card2, textvariable=self.sep_val, width=90)
        self.sep_entry.grid(row=2, column=0, sticky="w", padx=14, pady=(6, 12))

        ctk.CTkLabel(card2, text="Encoding (optional)", text_color="#A7B0C0").grid(row=1, column=1, sticky="w", padx=14)
        self.enc_entry = ctk.CTkEntry(card2, textvariable=self.encoding_val)
        self.enc_entry.grid(row=2, column=1, sticky="ew", padx=14, pady=(6, 12))

        self.run_btn = ctk.CTkButton(left, text="Clean & Analyze", height=44, corner_radius=16, command=self.run_cleaning)
        self.run_btn.grid(row=2, column=0, sticky="ew", padx=14, pady=(6, 10))

        self.status_lbl = ctk.CTkLabel(left, text="Ready.", text_color="#A7B0C0")
        self.status_lbl.grid(row=3, column=0, sticky="w", padx=18, pady=(0, 6))

        self.progress = ctk.CTkProgressBar(left, height=10, corner_radius=10)
        self.progress.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 10))
        self.progress.set(0)

        card3 = ctk.CTkFrame(left, corner_radius=16)
        card3.grid(row=5, column=0, sticky="ew", padx=14, pady=(0, 14))
        card3.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(card3, text="Quick actions", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=14, pady=(12, 8)
        )

        self.open_out_btn = ctk.CTkButton(card3, text="Open output folder", corner_radius=14, command=self.open_outdir, state="disabled")
        self.open_out_btn.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))

        self.open_cleaned_btn = ctk.CTkButton(card3, text="Open cleaned CSV", corner_radius=14, command=self.open_cleaned, state="disabled")
        self.open_cleaned_btn.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 8))

        self.open_summary_btn = ctk.CTkButton(card3, text="Open summary report", corner_radius=14, command=self.open_summary, state="disabled")
        self.open_summary_btn.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 14))

    def _build_right_panel(self):
        right = ctk.CTkFrame(self, corner_radius=18)
        right.grid(row=1, column=1, sticky="nsew", padx=(10, 16), pady=(0, 16))
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self.view_var = tk.StringVar(value="Summary")
        seg = ctk.CTkSegmentedButton(
            right, values=["Summary", "Preview", "Visuals"], variable=self.view_var, command=self._switch_view
        )
        seg.grid(row=0, column=0, sticky="w", padx=16, pady=(16, 10))

        self.view_container = ctk.CTkFrame(right, corner_radius=16)
        self.view_container.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        self.view_container.grid_rowconfigure(0, weight=1)
        self.view_container.grid_columnconfigure(0, weight=1)

        self.summary_view = self._build_summary_view(self.view_container)
        self.preview_view = self._build_preview_view(self.view_container)
        self.visuals_view = self._build_visuals_view(self.view_container)

        self._switch_view("Summary")

    def _build_summary_view(self, parent):
        frame = ctk.CTkFrame(parent, corner_radius=14)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self.summary_text = tk.Text(frame, wrap="word", relief="flat", bd=0, font=("Consolas", 10))
        self.summary_text.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self._apply_text_theme()

        return frame

    def _apply_text_theme(self):
        mode = ctk.get_appearance_mode().lower()
        if mode == "light":
            bg = "#F6F7FB"
            fg = "#111827"
        else:
            bg = "#0C1327"
            fg = "#E7EAF3"
        self.summary_text.configure(bg=bg, fg=fg, insertbackground=fg)

    def _build_preview_view(self, parent):
        frame = ctk.CTkFrame(parent, corner_radius=14)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        wrap = tk.Frame(frame, bd=0, highlightthickness=0)
        wrap.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        wrap.grid_rowconfigure(0, weight=1)
        wrap.grid_columnconfigure(0, weight=1)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", rowheight=26)
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

        self.tree = ttk.Treeview(wrap, show="headings")
        self.tree.grid(row=0, column=0, sticky="nsew")

        yscroll = ttk.Scrollbar(wrap, orient="vertical", command=self.tree.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=yscroll.set)

        return frame

    def _build_visuals_view(self, parent):
        frame = ctk.CTkFrame(parent, corner_radius=14)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self.visuals_frame = tk.Frame(frame, bd=0, highlightthickness=0)
        self.visuals_frame.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        return frame

    def _switch_view(self, _val=None):
        v = self.view_var.get()

        self.summary_view.grid_remove()
        self.preview_view.grid_remove()
        self.visuals_view.grid_remove()

        if v == "Summary":
            self.summary_view.grid()
        elif v == "Preview":
            self.preview_view.grid()
        else:
            self.visuals_view.grid()

        self._apply_text_theme()

    def browse_csv(self):
        path = filedialog.askopenfilename(
            title="Select CSV",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if path:
            self.input_path.set(path)

    def browse_outdir(self):
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.outdir_path.set(path)

    def set_busy(self, busy: bool):
        if busy:
            self.run_btn.configure(state="disabled")
            self.progress.start()
            self.status_lbl.configure(text="Running… cleaning + analysis in progress")
            self._disable_quick_actions()
        else:
            self.run_btn.configure(state="normal")
            self.progress.stop()
            self.progress.set(1)

    def run_cleaning(self):
        in_path = self.input_path.get().strip()
        if not in_path or not os.path.exists(in_path):
            messagebox.showerror("Missing CSV", "Please select a valid CSV file.")
            return

        outdir = self.outdir_path.get().strip()
        if not outdir:
            messagebox.showerror("Missing output folder", "Please choose an output folder.")
            return

        sep = self.sep_val.get().strip() or ","
        enc = self.encoding_val.get().strip() or None

        self.set_busy(True)

        self.summary_text.delete("1.0", "end")
        self.summary_text.insert(
            "end",
            "Running…\n• Cleaning messy values\n• Fixing types\n• Removing duplicates\n• Generating summary + charts\n",
        )
        self.view_var.set("Summary")
        self._switch_view()

        t = threading.Thread(target=self._worker, args=(in_path, outdir, sep, enc), daemon=True)
        t.start()

    def _worker(self, in_path, outdir, sep, enc):
        try:
            result = clean_and_analyze(in_path, outdir, sep=sep, encoding=enc)
            self.after(0, lambda: self._render_results(result))
        except Exception as e:
            self.after(0, lambda: self._show_error(e))

    def _show_error(self, e):
        self.set_busy(False)
        self.status_lbl.configure(text="Error.")
        messagebox.showerror("Error", f"Something went wrong:\n\n{e}")

    def _render_results(self, result):
        self.set_busy(False)

        df = result["df"]
        chart_path = result["chart_path"]

        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("end", result["report_text"])
        self.summary_text.insert("end", "\n\nFILES SAVED\n")
        self.summary_text.insert("end", f"- Cleaned CSV: {result['cleaned_path']}\n")
        self.summary_text.insert("end", f"- Summary TXT: {result['summary_path']}\n")
        self.summary_text.insert("end", f"- Charts PNG:  {chart_path}\n")

        self._load_treeview(df.head(80))
        self._draw_visuals(result, chart_path)

        self.last_outputs["cleaned"] = result["cleaned_path"]
        self.last_outputs["summary"] = result["summary_path"]
        self.last_outputs["chart"] = chart_path
        self.last_outputs["outdir"] = result["outdir"]
        self._enable_quick_actions()

        self.status_lbl.configure(text="Done ✅ Outputs generated successfully.")

    def _disable_quick_actions(self):
        for b in [self.open_out_btn, self.open_cleaned_btn, self.open_summary_btn]:
            b.configure(state="disabled")

    def _enable_quick_actions(self):
        for b in [self.open_out_btn, self.open_cleaned_btn, self.open_summary_btn]:
            b.configure(state="normal")

    def open_outdir(self):
        outdir = self.last_outputs.get("outdir")
        if outdir and os.path.isdir(outdir):
            open_path(outdir)

    def open_cleaned(self):
        p = self.last_outputs.get("cleaned")
        if p and os.path.exists(p):
            open_path(p)

    def open_summary(self):
        p = self.last_outputs.get("summary")
        if p and os.path.exists(p):
            open_path(p)

    def _load_treeview(self, df_preview):
        for c in self.tree.get_children():
            self.tree.delete(c)

        cols = list(df_preview.columns)
        self.tree["columns"] = cols

        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150, anchor="w")

        for _, row in df_preview.iterrows():
            vals = []
            for v in row.values:
                if pd.isna(v):
                    vals.append("")
                else:
                    s = str(v)
                    if len(s) > 75:
                        s = s[:72] + "..."
                    vals.append(s)
            self.tree.insert("", "end", values=vals)

    def _draw_visuals(self, result, chart_path):
        for w in self.visuals_frame.winfo_children():
            w.destroy()

        missing = result["missing_series"]
        numeric_cols = result["numeric_cols"]
        df = result["df"]

        fig = Figure(figsize=(10, 6), dpi=110)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.set_title("Missing Values (Top Columns)")
        if len(missing) == 0:
            ax1.text(0.5, 0.5, "No missing values 🎉", ha="center", va="center")
            ax1.set_axis_off()
        else:
            ax1.bar(missing.index.astype(str), missing.values)
            ax1.tick_params(axis="x", rotation=25)

        ax2.set_title("Histogram (first numeric column)")
        if len(numeric_cols) == 0:
            ax2.text(0.5, 0.5, "No numeric columns found", ha="center", va="center")
            ax2.set_axis_off()
        else:
            col = numeric_cols[0]
            data = df[col].dropna()
            ax2.hist(data, bins=30)
            ax2.set_xlabel(col)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.visuals_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        try:
            fig.savefig(chart_path, dpi=170, bbox_inches="tight")
        except Exception:
            pass


if __name__ == "__main__":
    app = CSVCleanerRounded()
    app.mainloop()
