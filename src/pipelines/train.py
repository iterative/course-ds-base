import argparse
import joblib
import pandas as pd
from typing import Text
import yaml

from src.train.train import train
from src.utils.logs import get_logger


def train_model(config_path: Text) -> None:
    """Train model.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('TRAIN', log_level=config['base']['log_level'])

    logger.info('Get estimator name')
    estimator_name = config['train']['estimator_name']
    logger.info(f'Estimator: {estimator_name}')

    logger.info('Load train dataset')
    train_df = pd.read_csv(config['data_split']['trainset_path'])

    logger.info('Train model')
    model = train(
        df=train_df,
        target_column=config['featurize']['target_column'],
        estimator_name=estimator_name,
        param_grid=config['train']['estimators'][estimator_name]['param_grid'],
        cv=config['train']['cv']
    )
    logger.info(f'Best score: {model.best_score_}')

    logger.info('Save model')
    models_path = config['train']['model_path']
    joblib.dump(model, models_path)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)
