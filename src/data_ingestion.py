import sys
import os

# Get the project's root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import pandas as pd
from config.db_config import get_mongo_connection

# Step 1: Connect to MongoDB
db = get_mongo_connection()
collection = db["cleaned_data"]  # Collection Name

# Step 2: Load the Cleaned CSV File
df = pd.read_csv("data/cleaned_data.csv")

# Step 3: Convert DataFrame to Dictionary
data_dict = df.to_dict("records")

# Step 4: Insert Data into MongoDB
collection.insert_many(data_dict)

print(f"✅ Successfully stored {len(data_dict)} records in MongoDB!")
