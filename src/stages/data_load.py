# Import Dependencies

import argparse
import pandas as pd
from sklearn.datasets import load_iris
from typing import Text
import yaml


# Load data functions

def data_load(config_file: Text) -> None:
    
    # Load configuration file
    with open('params.yaml') as conf_file:
        config = yaml.safe_load(conf_file)
    
    # load the raw data functions from sklearn
    data = load_iris(as_frame=True)
    dataset = data.frame
    
    # feature names curated from dataset
    dataset.columns = [colname.strip(' (cm)').replace(' ', '_') for colname in dataset.columns.tolist()]
       
    # Save raw data to path contained in params.yaml   
    dataset.to_csv(config['data_load']['dataset_csv'], index=False)
    
print ("data load completed successfully")

# Call the argparser api

if __name__ == '__main__':
    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", dest = 'config',required=True,help="input config file path")
    args = arg_parser.parse_args()
    
    data_load(config_file=args.config)