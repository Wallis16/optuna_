"""<>"""
import pandas as pd
from pymongo import MongoClient

def retrieve_from_mongodb(username: str, password: str, database: str,
                     collection_name: str):
    """<>"""
    mongo_uri = f'mongodb+srv://{username}:{password}@{database}.hi7evkw.mongodb.net/'

    client = MongoClient(mongo_uri)
    db = client[database]
    collection = db[collection_name]

    documents = collection.find()

    return pd.DataFrame(documents)
