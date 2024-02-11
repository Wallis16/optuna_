"""<>"""
import os
from dotenv import load_dotenv
from studies.optuna_sklearn import sklearn_model, sklearn_model_acc_time
from data.load import etl
from keys.data_extraction_keys import MongoConnection

load_dotenv()

username = os.getenv(MongoConnection.MONGODB_USER)
password = os.getenv(MongoConnection.MONGODB_PASSWORD)
database = os.getenv(MongoConnection.MONGODB_DATABASE)
collection_name = os.getenv(MongoConnection.MONGODB_COLLECTION)

numerical_features = ['passenger_count', 'trip_distance', 'payment_type', 'fare_amount']
directions = ['maximize','minimize']
N_TRIALS = 20

def main():
    """<>"""
    x,y = etl(username, password, database, collection_name, numerical_features)
    params = sklearn_model_acc_time(x, y, directions, N_TRIALS)
    return params
if __name__== "__main__":
    print(main())
