import pandas as pd
import ast
from pathlib import Path

DATA_DIR = Path("data/ptb-xl-1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")

def load_metadata():
    df = pd.read_csv(DATA_DIR / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    return df

def load_scp_statements():
    return pd.read_csv(DATA_DIR / "scp_statements.csv", index_col=0)

if __name__ == "__main__":
    df = load_metadata()
    scp = load_scp_statements()
    print(f"Records: {len(df)}")
    print(f"Patients: {df['patient_id'].nunique()}")
    print(f"SCP statements: {len(scp)}")
    print(df.head(2))