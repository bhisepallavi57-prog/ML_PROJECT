# This file generates data drift report using Evidently AI

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

def get_engine():
    # ✅ PostgreSQL connection (FIXED)
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)

def generate_drift_report():
    engine = get_engine()

    # ✅ load data from PostgreSQL
    df = pd.read_sql("SELECT * FROM transformed_shoppers", engine)

    if df.empty:
        print("❌ Table is empty")
        return

    # split data
    reference = df.iloc[:int(len(df) * 0.7)]
    current = df.iloc[int(len(df) * 0.7):]

    print(f"Reference size: {len(reference)} rows")
    print(f"Current size: {len(current)} rows")

    # create report
    report = Report(metrics=[DataDriftPreset()])

    # run report
    report.run(reference_data=reference, current_data=current)

    # save report
    os.makedirs('./monitoring', exist_ok=True)
    report_path = './monitoring/drift_report.html'
    report.save_html(report_path)

    print(f"✅ Drift report saved to {report_path}")

if __name__ == "__main__":
    generate_drift_report()