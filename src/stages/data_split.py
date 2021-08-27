import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Text
import yaml

from src.utils.logs import get_logger


def data_split(config_path: Text) -> None:
    """Split dataset into train/test.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('DATA_SPLIT', log_level=config['base']['log_level'])

    logger.info('Load features')
    dataset = pd.read_csv(config['featurize']['features_path'])

    logger.info('Split features into train and test sets')
    train_dataset, test_dataset = train_test_split(
        dataset,
        test_size=config['data_split']['test_size'],
        random_state=config['base']['random_state']
    )

    logger.info('Save train and test sets')
    train_csv_path = config['data_split']['trainset_path']
    test_csv_path = config['data_split']['testset_path']
    train_dataset.to_csv(train_csv_path, index=False)
    test_dataset.to_csv(test_csv_path, index=False)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_split(config_path=args.config)
