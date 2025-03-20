from config.db_config import get_mongo_connection
from src.transform import transform_data
from src.extract import extract_data

def load_data(df):
    """Loads transformed data into MongoDB."""
    db = get_mongo_connection()
    collection = db["transformed_supply_chain"]

    # Convert DataFrame to dictionary format for MongoDB
    data = df.to_dict(orient="records")
    
    # Insert into MongoDB
    collection.insert_many(data)

    print(f"✅ Loaded {len(data)} records into transformed_supply_chain collection")

if __name__ == "__main__":

    df = extract_data()  # Extract data
    transformed_df = transform_data(df)  # Transform data
    load_data(transformed_df)  # Load data
