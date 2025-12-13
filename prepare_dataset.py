import os
from pathlib import Path

import pandas as pd


WEAK_16K_DIR = Path(
    "/home/vlad/DESED_task/data/dcase/dataset/audio/train/weak_16k"
)
WEAK_TSV_PATH = Path(
    "/home/vlad/DESED_task/data/dcase/dataset/metadata/train/weak.tsv"
)


def main() -> None:
    df = pd.read_csv(WEAK_TSV_PATH, sep="\t")

    weak_files = {p.name for p in WEAK_16K_DIR.iterdir() if p.is_file()}

    missing_files = []
    for _, row in df.iterrows():
        if row.filename not in weak_files:
            print(f"missing file: {row.filename}")
            missing_files.append(row.filename)

    df_clean = df[~df["filename"].isin(missing_files)]

    df_clean.to_csv(
        WEAK_TSV_PATH,
        sep="\t",
        index=False,
    )


if __name__ == "__main__":
    main()
