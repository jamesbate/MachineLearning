from __future__ import print_function
import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset
#preamble
########################################################################
#Display Settings
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
#Read in Data
chd = pd.read_csv("california_housing_train.csv",sep=",")
chd=chd.reindex(np.random.permutation(chd.index))
chd["median_house_value"] /= 1000.0
#separate final 10 entries to test the trained model
targets = chd[:-10]["median_house_value"]
test_targets = chd[-10:]["median_house_value"]

feature_train = chd[["total_rooms"]][:-10]
feature_test = chd[["total_rooms"]][-10:]
feature_columns = [tf.feature_column.numeric_column("total_rooms")]



#Set Learning Model
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0000001)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer,5.0)

linear_regressor = tf.estimator.LinearRegressor(
        feature_columns = feature_columns,
        optimizer = optimizer
        )

#this function effectively controls how to feed the linear regressor
def input_function(features,targets,batch_size=1,shuffle=True,num_epochs=None):
    """Trains a linear regression model of one feature.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    features = {key:np.array(value) for key,value in dict(features).items()}
    
    ds = Dataset.from_tensor_slices((features,targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds = ds.shuffle(buffer_size = 10000)
    features,labels = ds.make_one_shot_iterator().get_next()
    return features,labels

_ = linear_regressor.train(input_fn=lambda:input_function(feature_train,targets),steps = 100)

#Now test the model 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    