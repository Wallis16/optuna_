"""<>"""
from data.extract import retrieve_from_mongodb
from data.transform import prepare_data

def etl(username, password, database, collection_name, features):
    """<>"""
    df = retrieve_from_mongodb(username, password, database, collection_name)
    x, y = prepare_data(data=df, numerical_features=features)
    return x, y
