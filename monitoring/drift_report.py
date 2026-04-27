import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

def get_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:"
        f"{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:"
        f"{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url)

def generate_drift_report():
    engine = get_engine()

    # Load data from PostgreSQL
    df = pd.read_sql("SELECT * FROM transformed_shoppers", engine)

    # Split data
    reference = df.iloc[:int(len(df) * 0.7)]
    current = df.iloc[int(len(df) * 0.7):]

    print(f"Reference size: {len(reference)} rows")
    print(f"Current size: {len(current)} rows")

    # Generate drift report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    os.makedirs("./monitoring", exist_ok=True)
    report_path = "./monitoring/drift_report.html"

    report.save_html(report_path)

    print("✅ Drift report saved successfully")

if __name__ == "__main__":
    generate_drift_report()