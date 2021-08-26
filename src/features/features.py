import pandas as pd


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features.
    Args:
        df {pandas.DataFrame}: dataset
    Returns:
        pandas.DataFrame: updated dataset with new features
    """

    dataset = df.copy()
    dataset['sepal_length_to_sepal_width'] = dataset['sepal_length'] / dataset['sepal_width']
    dataset['petal_length_to_petal_width'] = dataset['petal_length'] / dataset['petal_width']

    dataset = dataset[[
        'sepal_length', 'sepal_width', 'petal_length', 'petal_width',
        'sepal_length_to_sepal_width', 'petal_length_to_petal_width',
        'target'
    ]]

    return dataset
