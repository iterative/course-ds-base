import argparse
import joblib
import json
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, f1_score
from typing import Text, Dict
import yaml

from src.report.visualize import plot_confusion_matrix
from src.utils.logs import get_logger

def convert_to_labels(indexes, labels):
    result = []
    for i in indexes:
        result.append(labels[i])
    return result

def write_confusion_matrix_data(y_true, predicted, labels, filename):
    assert len(predicted) == len(y_true)
    predicted_labels = convert_to_labels(predicted, labels)
    true_labels = convert_to_labels(y_true, labels)
    cf = pd.DataFrame(list(zip(true_labels, predicted_labels)), columns=["y_true", "predicted"])
    cf.to_csv(filename, index=False)


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
    target_column=config['featurize']['target_column']
    y_test = test_df.loc[:, target_column].values
    X_test = test_df.drop(target_column, axis=1).values

    prediction = model.predict(X_test)
    f1 = f1_score(y_true=y_test, y_pred=prediction, average='macro')

    labels = load_iris(as_frame=True).target_names.tolist()
    cm = confusion_matrix(y_test, prediction)

    report = {
        'f1': f1,
        'cm': cm,
        'actual': y_test,
        'predicted': prediction
    }

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
                                target_names=labels,
                                normalize=False)
    confusion_matrix_png_path = reports_folder / config['evaluate']['confusion_matrix_image']
    plt.savefig(confusion_matrix_png_path)
    logger.info(f'Confusion matrix saved to : {confusion_matrix_png_path}')

    confusion_matrix_data_path = reports_folder / config['evaluate']['confusion_matrix_data']
    write_confusion_matrix_data(y_test, prediction, labels=labels, filename=confusion_matrix_data_path)
    logger.info(f'Confusion matrix data saved to : {confusion_matrix_data_path}')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate_model(config_path=args.config)
