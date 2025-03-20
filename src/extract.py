import pandas as pd
from config.db_config import get_mongo_connection

def extract_data():
    db = get_mongo_connection()
    collection = db["cleaned_data"]  # Your collection name
    data = list(collection.find({}, {"_id": 0}))  # Ignore MongoDB ID field
    
    df = pd.DataFrame(data)
    print(f"✅ Extracted {df.shape[0]} rows from MongoDB")
    return df

if __name__ == "__main__":
    extract_data()
