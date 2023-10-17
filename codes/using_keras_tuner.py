"""Bayesian Optimization with Cross-Validation"""
from pandas import read_csv, DataFrame                                          
from sklearn.model_selection import train_test_split                            
from sklearn import model_selection, metrics
from tensorflow.keras import Sequential, optimizers                             
from tensorflow.keras.layers import Dense                                       
from tensorflow.keras.utils import normalize                                    
from tensorflow.keras.callbacks import EarlyStopping                            
from tensorflow import keras
import tensorflow as tf                                                         
import numpy as np                                                              
import matplotlib.pyplot as plt   
import kerastuner as kt

# load data from github repository
path =\
'https://raw.githubusercontent.com/shindj91/data_set/master/data_segr_11_unique.csv'
data_set = read_csv(path, header=0)                                                   
# split into input and output
features = list(data_set.columns)[:-1]
X_framed, y_framed = data_set.loc[:, features], data_set.loc[:, 'Esegr']
# Max-min normalization only for X
X_scaled = ((X_framed-X_framed.min())/(X_framed.max()-X_framed.min())) 
X, y = X_scaled.values, y_framed.values
# split into train and test datasets                                            
print(X.shape, y.shape)  
n_features = X.shape[1]

epochs = 40000
max_trials = 200
directory = './210518_keras_tuner/'

def model_builder(hp):
  hp_lambda = hp.Float('l2', 0.0001, 0.1, step=0.001)
  hp_learning_rate = hp.Float('learning_rate', 0.001, 0.5, step=0.001) 
  hp_activation = hp.Choice('activation', values = ['relu', 'elu', 'sigmoid', 'tanh'])
  # Define a NN model
  model = Sequential()
  model.add(Dense(units = hp.Int('units_1', 1, 100, step=1),
                  activation = hp_activation,
                  kernel_initializer='he_normal',
                  kernel_regularizer=tf.keras.regularizers.l2(hp_lambda),
                  input_shape=(n_features,)))
  model.add(Dense(units = hp.Int('units_2', 1, 100, step=1),
                  activation = hp_activation,
                  kernel_initializer='he_normal',
                  kernel_regularizer=tf.keras.regularizers.l2(hp_lambda)))
  model.add(Dense(units = hp.Int('units_3', 1, 100, step=1),
                  activation = hp_activation,
                  kernel_initializer='he_normal',
                  kernel_regularizer=tf.keras.regularizers.l2(hp_lambda)))
  model.add(Dense(units = hp.Int('units_4', 1, 100, step=1),
                  activation = hp_activation,
                  kernel_initializer='he_normal',
                  kernel_regularizer=tf.keras.regularizers.l2(hp_lambda)))

  
  model.add(Dense(1)) # linear activation
  model.compile(optimizer = keras.optimizers.Adam(
      learning_rate = hp_learning_rate),
                loss = 'mse', 
                metrics = ['mae'])
  return model

# Source: https://github.com/keras-team/keras-tuner/issues/122
class CVTuner(kt.engine.tuner.Tuner):
  def run_trial(self, trial, x, y, callbacks, batch_size=32,  epochs=1, verbose=0):
    cv = model_selection.KFold(n_splits=10, shuffle=True)
    val_maes = []
    for train_indices, test_indices in cv.split(x):
      x_train, x_test = x[train_indices], x[test_indices]
      y_train, y_test = y[train_indices], y[test_indices]
      model = self.hypermodel.build(trial.hyperparameters)
      model.fit(x_train, y_train,
                validation_data=(x_test, y_test),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=verbose)
      _, val_mae = model.evaluate(x_test, y_test)
      # mse and mae for validation set will be printed.
      val_maes.append(val_mae)
    self.oracle.update_trial(trial.trial_id, {'val_cv_mae': np.mean(val_maes)})
    self.save_model(trial.trial_id , model)

# Defining callback
early_stopping = EarlyStopping(monitor='mae',
                               patience=1000,
                               restore_best_weights=True)

# Make a CVTuner instance
tuner = CVTuner(
    oracle=kt.oracles.BayesianOptimization(
        objective=kt.Objective("val_cv_mae", direction="min"),
        max_trials=max_trials),
    hypermodel=model_builder,
    directory=directory,
    project_name='keras_tuner_cv',
    overwrite=True) # scoring=metrics.make_scorer(metrics.accuracy_score),
# For kerastuner v1.0.1, the objective above is used as the score for Bayesian
# opt.
tuner.search_space_summary()
tuner.search(X, y, epochs = epochs, 
        batch_size=256,
        callbacks = [early_stopping]) 

# best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
# best_model = tuner.get_best_models(num_models=1)[0]

# # tuner.save_model(2, best_model)
# best_model.save(tuner.directory+'tuned_model_210410.h5')
# best_model.summary()

# make a prediction                                                         
#yhat= best_model.predict(X)
#plt.title('Parity plot for segregation energy')                        
#plt.xlabel('y (eV)')                                                  
#plt.ylabel('y_predicted (eV)')                                       
#plt.scatter(y, yhat)                                                
#bottom, top = plt.xlim()                                           
#plt.ylim(bottom, top)

# # Build the model with the optimal hyperparameters and train it on the data
# model = tuner.hypermodel.build(best_hps)
# model.fit(X_train, y_train, epochs = epochs, validation_data = (X_test, y_test))
# model.save(tuner.directory+'tuned_model_trained.h5')
