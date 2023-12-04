# Install Packages and Dependencies

import tensorflow as tf
import tensorflow_data_validation as tfdv
import pandas as pd

from sklearn.model_selection import train_test_split
from util import add_extra_rows

from tensorflow_metadata.proto.v0 import schema_pb2

print('TFDV Version: {}'.format(tfdv.__version__))
print('Tensorflow Version: {}'.format(tf.__version__))

# Unpack the dataset the "data/aduclt.data" directory

# Read in the training and evaluation datasets
df = pd.read_csv('data/adult.data', skipinitialspace=True)

# Split the dataset. Do not shuffle for this demo notebook.
train_df, eval_df = train_test_split(df, test_size=0.2, shuffle=False)

# Stopped code here for testing.