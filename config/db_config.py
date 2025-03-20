from pymongo import MongoClient

def get_mongo_connection():
    """
    Connect to MongoDB and return the database object.
    """
    client = MongoClient("mongodb://localhost:27017/")  # Connect to local MongoDB
    db = client["supply_chain_db"]  # Create/Get database
    return db
