from src.extract import extract_data
from src.transform import transform_data
from src.load import load_data

def run_etl():
    print("\n🚀 Running ETL Pipeline...\n")

    # Extract - Run only once!
    df = extract_data()

    # Transform - Pass extracted data
    transformed_df = transform_data(df)

    # Load - Pass transformed data
    load_data(transformed_df)

    print("\n✅ ETL Pipeline Completed!")

if __name__ == "__main__":
    run_etl()
