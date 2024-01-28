import argparse
import pandas as pd
from typing import Text
import yaml

from src.utils.logs import get_logger


def featurize(config_path: Text) -> None:
    """Create new features.
    Args:
        config_path {Text}: path to config
    """

    with open('params.yaml') as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('FEATURIZE', log_level=config['base']['log_level'])

    logger.info('Load the raw data')
    dataset = pd.read_csv(config['data_load']['dataset_csv'])

    logger.info('Curate by extraction of features from the dataset')
    dataset['sepal_length_to_sepal_width'] = dataset['sepal_length'] / dataset['sepal_width']
    dataset['petal_length_to_petal_width'] = dataset['petal_length'] / dataset['petal_width']
    featured_dataset = dataset[[
        'sepal_length', 'sepal_width', 'petal_length', 'petal_width',
        'sepal_length_to_sepal_width', 'petal_length_to_petal_width',
        'target'
    ]]

    logger.info('Save features')
    features_path = config['featurize']['features_path']
    featured_dataset.to_csv(features_path, index=False)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True, help="curate dataset")
    args = args_parser.parse_args()

    featurize(config_path=args.config)