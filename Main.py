"""
This file train and test an LSTM model with the stored data.

"""

import tensorflow as tf
from sklearn import preprocessing
import pickle
import TrainTicker
import numpy as np
#%% Import Data
# Highly-correlated data is captured
with open('All_corr.pickle', 'rb') as f:
    Top_Bottom_Five_Correlated_Names = pickle.load(f)

# Price Data is captured
with open('All_1d.pickle', 'rb') as f:
    x = pickle.load(f)
x = x[:-5]  # remove the duplicates

# Trading Volume Data is captured
with open('All_1dv.pickle', 'rb') as f:
    v = pickle.load(f)
#%% Which Ticker you want to use?
ticker = "MSFT"
print(Top_Bottom_Five_Correlated_Names[ticker])
D = TrainTicker.MyData(ticker, x, v, Top_Bottom_Five_Correlated_Names)

#%% Extract data

history_points = 100    # How far to remember?
forward_days = 10   # I will only use 1 (the next day)

data_normaliser = preprocessing.MinMaxScaler()
data_normalised = data_normaliser.fit_transform(D)
D2 = np.array(D)

# using the last {history_points} data points, predict the next Adj close value
histories_normalised = np.array([data_normalised[i: i + history_points].copy() for i in range(len(data_normalised) - history_points - forward_days)])
prediction_normalised = np.array([data_normalised[i + history_points: i + history_points+forward_days, 0].copy() for i in range(len(data_normalised) - history_points - forward_days)])
prediction_normalised = np.expand_dims(prediction_normalised, -1)

# histories= np.array([D2[i: i + history_points, 0].copy() for i in range(len(D2) - history_points - forward_days)])
prediction = np.array([D2[i + history_points: i + history_points + forward_days, 0].copy() for i in range(len(D2) - history_points - forward_days)])
# prediction = np.expand_dims(prediction, -1)
y_normaliser = preprocessing.MinMaxScaler()
y_normaliser.fit(np.expand_dims(D2[:, 0], -1))
#%% Prepare data
test_split = 0.9    # the percent of data to be used for training
n = int(histories_normalised.shape[0] * test_split)

# splitting the dataset up into train and test sets

Xtrain = histories_normalised[:n]
Ytrain = prediction_normalised[:n]

Xtest = histories_normalised[n:]
Ytest = prediction_normalised[n:]

unscaled_y_test = prediction[n:]
# unscaled_y_train = histories[:n]
#%% Build the model
NUM_NEURONS_FirstLayer = history_points
NUM_NEURONS_SecondLayer = 64
EPOCHS = 50

print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(history_points,17), name='In'))
model.add(tf.keras.layers.LSTM(NUM_NEURONS_FirstLayer, return_sequences=True, name='l1'))
model.add(tf.keras.layers.Dropout(.2, name='D1'))
model.add(tf.keras.layers.LSTM(NUM_NEURONS_SecondLayer, name='l2'))
# model.add(tf.keras.layers.Dense(64, name='FC1'))
# model.add(tf.keras.layers.Activation('sigmoid'))
model.add(tf.keras.layers.Dense(1, name='FC2'))
# model.add(tf.keras.layers.Activation('linear', name='Out'))

opt = tf.keras.optimizers.Adam(lr=0.0005)
model.compile(loss='mean_squared_error', optimizer=opt)
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
#%% Training the model

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./tensorboard')
# Only the next day value as the target
XX = Xtrain[:, :, 0]
XX = np.expand_dims(XX, -1)
YY = Ytrain[:, 0, :]
# YY = np.expand_dims(YY, -1)
history = model.fit(Xtrain, YY,
                    epochs=EPOCHS,
                    shuffle=True,
                    batch_size=32,
                    callbacks=tensorboard,
                    verbose=2)
# evaluation = model.evaluate(Xtest[:,:,0], Ytest[:,0,:])
# print(evaluation)
#%% Prediction
y_test_predicted1 = model.predict(Xtest)
# model.predict returns normalised values
# now we scale them back up using the y_scaler from before
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted1)

# also getting predictions for the entire dataset, just to see how it performs
y_predicted = model.predict(Xtrain)
y_predicted = y_normaliser.inverse_transform(y_predicted)

real_mse_test = np.mean(np.square(unscaled_y_test[:,0] - y_test_predicted))
scaled_mse_test = real_mse_test / (np.max(unscaled_y_test[:,0]) - np.min(unscaled_y_test[:,0])) * 100
print(scaled_mse_test)

y_real = D2[100:n+100,0]
real_mse_train = np.mean(np.square(y_real - np.squeeze(y_predicted)))
scaled_mse_train = real_mse_train / (np.max(y_real) - np.min(y_real)) * 100
print(scaled_mse_train)

#%% Plot
import matplotlib.pyplot as plt
plt.gcf().set_size_inches(22, 15, forward=True)


start = 0
end = -1

real = plt.plot(y_real, label='real')
pred = plt.plot(y_predicted[start:end,0], label='predicted')

# real = plt.plot(unscaled_y_test[start:end, 0], label='real')
# pred = plt.plot(y_test_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted'])

plt.show()

#%%

