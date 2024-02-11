"""<>"""
from sklearn.model_selection import train_test_split
def prepare_data(data, numerical_features):
    """<>"""
    data = data[numerical_features]

    data = data.dropna()

    x, y = data[numerical_features[:-1]], data[numerical_features[-1]]

    x_train, _, y_train, _ = train_test_split(x, y.values.ravel(), test_size=0.33, random_state=42)

    return x_train, y_train
