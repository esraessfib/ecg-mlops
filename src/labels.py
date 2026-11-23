"""
Label extraction for PTB-XL multi-label classification.

Reads raw SCP codes from the database, keeps only diagnostic statements,
and maps them to the five superclasses: NORM, MI, STTC, CD, HYP.
"""

import pandas as pd
import ast
from pathlib import Path

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def build_label_matrix(df: pd.DataFrame, scp: pd.DataFrame) -> pd.DataFrame:
    """
    Build a binary label matrix for the five diagnostic superclasses.

    Args:
        df:  metadata DataFrame with a parsed 'scp_codes' column (dict per row)
        scp: scp_statements DataFrame with 'diagnostic' and 'diagnostic_class' columns

    Returns:
        DataFrame indexed by ecg_id with one binary column per superclass.
        A record receives label 1 for a superclass if it has at least one
        diagnostic SCP code belonging to that superclass.
    """
    # Keep only the SCP codes that are diagnostic statements
    diagnostic_codes = scp[scp["diagnostic"] == 1.0]["diagnostic_class"]

    labels = []

    for ecg_id, row in df.iterrows():
        record_labels = {cls: 0 for cls in SUPERCLASSES}

        for code in row["scp_codes"]:
            # Skip codes that are not diagnostic (e.g. rhythm statements like SR)
            if code not in diagnostic_codes.index:
                continue

            superclass = diagnostic_codes[code]

            if superclass in record_labels:
                record_labels[superclass] = 1

        labels.append({"ecg_id": ecg_id, **record_labels})

    return pd.DataFrame(labels).set_index("ecg_id")


def filter_labeled_records(
    df: pd.DataFrame, label_matrix: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove records that have no superclass label at all.

    Some records only contain rhythm or form codes with no diagnostic
    superclass, making them unusable for classification.

    Returns:
        Tuple of (filtered metadata DataFrame, filtered label matrix).
    """
    has_label = label_matrix.sum(axis=1) > 0
    return df[has_label], label_matrix[has_label]


if __name__ == "__main__":
    from data_loader import load_metadata, load_scp_statements

    df = load_metadata()
    scp = load_scp_statements()

    label_matrix = build_label_matrix(df, scp)
    df_filtered, labels_filtered = filter_labeled_records(df, label_matrix)

    print(f"Total records:          {len(df)}")
    print(f"Records with a label:   {len(df_filtered)}")
    print(f"\nLabel counts per superclass:")
    print(labels_filtered.sum().to_string())
    print(f"\nMulti-label records (more than one superclass):")
    print((labels_filtered.sum(axis=1) > 1).sum())