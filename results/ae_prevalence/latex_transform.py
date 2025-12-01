import pandas as pd
from pathlib import Path

# the CSVs are in the current directory
input_dir = Path.cwd()
# Where to put the generated .tex files (can be same as input_dir)
output_dir = input_dir
output_dir.mkdir(parents=True, exist_ok=True)

for csv_path in sorted(input_dir.glob("*.csv")):
    df = pd.read_csv(csv_path)

    # Drop significant, drug1_name, drug2_name, drug3_name columns if they exist
    cols_to_drop = [col for col in df.columns if col.startswith("drug") and col.endswith("_name")]
    if "significant" in df.columns:
        cols_to_drop.append("significant")
    df = df.drop(columns=cols_to_drop, errors="ignore").head(20)  # limit to first 10 rows

    # Basic LaTeX table; tweak column_format as needed
    latex_table = df.to_latex(
        index=False,          # don't show the DataFrame index
        escape=True,          # escape LaTeX special chars in the data
        longtable=False,      # set True if you want longtable
        bold_rows=False,      # don't bold the header row labels
        column_format=None,   # e.g., "lrrr" if you want manual alignment
         float_format="%.2e",  # <-- scientific notation, 2 decimal places
    )
    
    # Wrap in a full table environment with caption/label templates
    table_env = rf"""
\begin{{table}}[htbp]
    \centering
    \caption{{Highest ae prop Opioid drug pairs (baseline 1.70e-02). Pvalue from chi-sq test and adjusted with Benjamini Hochberg. Drugs in ATC3 class codes. Filtered to top 20 significant rows with n_total >200}}
    \vspace{{6pt}}
    \label{{tab:{csv_path.stem}}}
{latex_table}
\end{{table}}
""".strip() + "\n"

    tex_path = output_dir / f"{csv_path.stem}.tex"
    tex_path.write_text(table_env)
    print(f"Wrote {tex_path}")
