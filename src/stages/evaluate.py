import argparse
import joblib
import json
import pandas as pd
from pathlib import Path
from typing import Text
import yaml

from src.data.dataset import get_target_names
from src.evaluate.evaluate import evaluate
from src.report.visualize import plot_confusion_matrix
from src.utils.logs import get_logger


def evaluate_model(config_path: Text) -> None:
    """Evaluate model.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('EVALUATE', log_level=config['base']['log_level'])

    logger.info('Load model')
    model_path = config['train']['model_path']
    model = joblib.load(model_path)

    logger.info('Load test dataset')
    test_df = pd.read_csv(config['data_split']['testset_path'])

    logger.info('Evaluate (build report)')
    report = evaluate(df=test_df,
                      target_column=config['featurize']['target_column'],
                      clf=model)

    logger.info('Save metrics')
    # save f1 metrics file
    reports_folder = Path(config['evaluate']['reports_dir'])
    metrics_path = reports_folder / config['evaluate']['metrics_file']

    json.dump(
        obj={'f1_score': report['f1']},
        fp=open(metrics_path, 'w')
    )

    logger.info(f'F1 metrics file saved to : {metrics_path}')

    logger.info('Save confusion matrix')
    # save confusion_matrix.png
    plt = plot_confusion_matrix(cm=report['cm'],
                                target_names=get_target_names(),
                                normalize=False)
    confusion_matrix_png_path = reports_folder / config['evaluate']['confusion_matrix_image']
    plt.savefig(confusion_matrix_png_path)
    logger.info(f'Confusion matrix saved to : {confusion_matrix_png_path}')


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate_model(config_path=args.config)
