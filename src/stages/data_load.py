import argparse
from typing import Text
import yaml

from src.data.dataset import get_dataset
from src.utils.logs import get_logger


def data_load(config_path: Text) -> None:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('DATA_LOAD', log_level=config['base']['log_level'])

    logger.info('Get dataset')
    dataset = get_dataset()

    logger.info('Save raw data')
    dataset.to_csv(config['data_load']['dataset_csv'], index=False)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)
