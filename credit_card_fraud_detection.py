# Setup

import tensorflow as tf
from tensorflow import keras

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Data processing and exploration

file = tf.keras.utils
raw_df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
raw_df.head()

# Examine the class label imbalance

neg, pos = np.bincount(raw_df['Class'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))

#Examples:
#    Total: 284807
#    Positive: 492 (0.17% of total)
#This shows the small fraction of positive samples.

# Clean, split and normalize the data

#The raw data has a few issues. First the Time and Amount columns are too variable to use directly.
# Drop the Time column (since it's not clear what it means) and take the log of the Amount column to reduce its range.

cleaned_df = raw_df.copy()

# You don't want the `Time` column.
cleaned_df.pop('Time')

# The `Amount` column covers a huge range. Convert to log-space.
eps=0.001 # 0 => 0.1¢
cleaned_df['Log Ammount'] = np.log(cleaned_df.pop('Amount')+eps)

#Split the dataset into train, validation, and test sets.
# The validation set is used during the model fitting to evaluate the loss and any metrics, however the model is not fit with this data.
# The test set is completely unused during the training phase and is only used at the end to evaluate how well the model generalizes to new data.
# This is especially important with imbalanced datasets where overfitting is a significant concern from the lack of training data.

# Use a utility from sklearn to split and shuffle our dataset.
train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('Class'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('Class'))
test_labels = np.array(test_df.pop('Class'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)

#Normalize the input features using the sklearn StandardScaler. This will set the mean to 0 and standard deviation to 1.

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)


print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)


#Training labels shape: (182276,)
#Validation labels shape: (45569,)
#Test labels shape: (56962,)
#Training features shape: (182276, 29)
#Validation features shape: (45569, 29)
#Test features shape: (56962, 29)

# Look at the data distribution

# Compare the distributions of the positive and negative examples over a few features.
# Good questions to ask yourself at this point are:

#Do these distributions make sense?
#Yes. You've normalized the input and these are mostly concentrated in the +/- 2 range.
#Can you see the difference between the distributions?
#Yes the positive examples contain a much higher rate of extreme values.
pos_df = pd.DataFrame(train_features[ bool_train_labels], columns = train_df.columns)
neg_df = pd.DataFrame(train_features[~bool_train_labels], columns = train_df.columns)

sns.jointplot(pos_df['V5'], pos_df['V6'],
              kind='hex', xlim = (-5,5), ylim = (-5,5))
plt.suptitle("Positive distribution")

sns.jointplot(neg_df['V5'], neg_df['V6'],
              kind='hex', xlim = (-5,5), ylim = (-5,5))
_ = plt.suptitle("Negative distribution")


# Define the model and metrics

#Define a function that creates a simple neural network with a densly connected hidden layer, a dropout layer to reduce overfitting, and an output sigmoid layer that returns the probability of a transaction being fraudulent:

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
]

def make_model(metrics = METRICS, output_bias=None):
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)
  model = keras.Sequential([
      keras.layers.Dense(
          16, activation='relu',
          input_shape=(train_features.shape[-1],)),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(1, activation='sigmoid',
                         bias_initializer=output_bias),
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(lr=1e-3),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)

  return model

# Build the model

EPOCHS = 100
BATCH_SIZE = 2048

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc',
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

model = make_model()
model.summary()

"""Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 16)                480       
_________________________________________________________________
dropout (Dropout)            (None, 16)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 17        
=================================================================
Total params: 497
Trainable params: 497
Non-trainable params: 0
_________________________________________________________________
"""

#Test run the model:

model.predict(train_features[:10])

"""
array([[0.43769827],
       [0.93449134],
       [0.43766674],
       [0.45128497],
       [0.8489004 ],
       [0.6685503 ],
       [0.64561343],
       [0.40402412],
       [0.4268172 ],
       [0.3361551 ]], dtype=float32)
"""

#Optional: Set the correct initial bias.

results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))

#Loss: 0.9491

initial_bias = np.log([pos/neg])
#array([-6.35935934])

model = make_model(output_bias = initial_bias)
model.predict(train_features[:10])
"""array([[0.0045591 ],
       [0.00154175],
       [0.00143366],
       [0.00461797],
       [0.00075015],
       [0.00152292],
       [0.00843064],
       [0.00139166],
       [0.00732469],
       [0.00100218]], dtype=float32)"""

results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))

#This initial loss is about 50 times less than if would have been with naive initialization.

#This way the model doesn't need to spend the first few epochs just learning that positive examples are unlikely.

#Checkpoint the initial weights

#To make the various training runs more comparable, keep this initial model's weights in a checkpoint file, and load them into each model before training.

initial_weights = os.path.join(tempfile.mkdtemp(),'initial_weights')
model.save_weights(initial_weights)

#Confirm that the bias fix helps

model = make_model()
model.load_weights(initial_weights)
model.layers[-1].bias.assign([0.0])
zero_bias_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(val_features, val_labels),
    verbose=0)

model = make_model()
model.load_weights(initial_weights)
careful_bias_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=20,
    validation_data=(val_features, val_labels),
    verbose=0)


def plot_loss(history, label, n):
    # Use a log scale to show the wide range of values.
    plt.semilogy(history.epoch, history.history['loss'],
                 color=colors[n], label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'],
                 color=colors[n], label='Val ' + label,
                 linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()


plot_loss(zero_bias_history, "Zero Bias", 0)
plot_loss(careful_bias_history, "Careful Bias", 1)

#Train the model

model = make_model()
model.load_weights(initial_weights)
baseline_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks = [early_stopping],
    validation_data=(val_features, val_labels))

"""
Epoch 1/100
90/90 [==============================] - 1s 13ms/step - loss: 0.0180 - tp: 85.0000 - fp: 323.0000 - tn: 227118.0000 - fn: 319.0000 - accuracy: 0.9972 - precision: 0.2083 - recall: 0.2104 - auc: 0.7760 - val_loss: 0.0077 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 45488.0000 - val_fn: 81.0000 - val_accuracy: 0.9982 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00 - val_auc: 0.9335
Epoch 2/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0089 - tp: 93.0000 - fp: 65.0000 - tn: 181888.0000 - fn: 230.0000 - accuracy: 0.9984 - precision: 0.5886 - recall: 0.2879 - auc: 0.8504 - val_loss: 0.0049 - val_tp: 38.0000 - val_fp: 6.0000 - val_tn: 45482.0000 - val_fn: 43.0000 - val_accuracy: 0.9989 - val_precision: 0.8636 - val_recall: 0.4691 - val_auc: 0.9495
Epoch 3/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0067 - tp: 143.0000 - fp: 33.0000 - tn: 181920.0000 - fn: 180.0000 - accuracy: 0.9988 - precision: 0.8125 - recall: 0.4427 - auc: 0.9075 - val_loss: 0.0042 - val_tp: 49.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 32.0000 - val_accuracy: 0.9991 - val_precision: 0.8750 - val_recall: 0.6049 - val_auc: 0.9501
Epoch 4/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0059 - tp: 173.0000 - fp: 33.0000 - tn: 181920.0000 - fn: 150.0000 - accuracy: 0.9990 - precision: 0.8398 - recall: 0.5356 - auc: 0.9051 - val_loss: 0.0039 - val_tp: 55.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 26.0000 - val_accuracy: 0.9993 - val_precision: 0.8871 - val_recall: 0.6790 - val_auc: 0.9503
Epoch 5/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0060 - tp: 180.0000 - fp: 37.0000 - tn: 181916.0000 - fn: 143.0000 - accuracy: 0.9990 - precision: 0.8295 - recall: 0.5573 - auc: 0.8989 - val_loss: 0.0037 - val_tp: 57.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 24.0000 - val_accuracy: 0.9993 - val_precision: 0.8906 - val_recall: 0.7037 - val_auc: 0.9503
Epoch 6/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0054 - tp: 184.0000 - fp: 42.0000 - tn: 181911.0000 - fn: 139.0000 - accuracy: 0.9990 - precision: 0.8142 - recall: 0.5697 - auc: 0.9139 - val_loss: 0.0035 - val_tp: 58.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 23.0000 - val_accuracy: 0.9993 - val_precision: 0.8923 - val_recall: 0.7160 - val_auc: 0.9503
Epoch 7/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0052 - tp: 182.0000 - fp: 36.0000 - tn: 181917.0000 - fn: 141.0000 - accuracy: 0.9990 - precision: 0.8349 - recall: 0.5635 - auc: 0.9158 - val_loss: 0.0034 - val_tp: 58.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 23.0000 - val_accuracy: 0.9993 - val_precision: 0.8923 - val_recall: 0.7160 - val_auc: 0.9503
Epoch 8/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0052 - tp: 185.0000 - fp: 35.0000 - tn: 181918.0000 - fn: 138.0000 - accuracy: 0.9991 - precision: 0.8409 - recall: 0.5728 - auc: 0.9161 - val_loss: 0.0033 - val_tp: 58.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 23.0000 - val_accuracy: 0.9993 - val_precision: 0.8923 - val_recall: 0.7160 - val_auc: 0.9504
Epoch 9/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0053 - tp: 178.0000 - fp: 31.0000 - tn: 181922.0000 - fn: 145.0000 - accuracy: 0.9990 - precision: 0.8517 - recall: 0.5511 - auc: 0.9084 - val_loss: 0.0032 - val_tp: 58.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 23.0000 - val_accuracy: 0.9993 - val_precision: 0.8923 - val_recall: 0.7160 - val_auc: 0.9503
Epoch 10/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0049 - tp: 184.0000 - fp: 30.0000 - tn: 181923.0000 - fn: 139.0000 - accuracy: 0.9991 - precision: 0.8598 - recall: 0.5697 - auc: 0.9118 - val_loss: 0.0031 - val_tp: 58.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 23.0000 - val_accuracy: 0.9993 - val_precision: 0.8923 - val_recall: 0.7160 - val_auc: 0.9565
Epoch 11/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0048 - tp: 185.0000 - fp: 36.0000 - tn: 181917.0000 - fn: 138.0000 - accuracy: 0.9990 - precision: 0.8371 - recall: 0.5728 - auc: 0.9226 - val_loss: 0.0031 - val_tp: 58.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 23.0000 - val_accuracy: 0.9993 - val_precision: 0.8923 - val_recall: 0.7160 - val_auc: 0.9565
Epoch 12/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0048 - tp: 181.0000 - fp: 33.0000 - tn: 181920.0000 - fn: 142.0000 - accuracy: 0.9990 - precision: 0.8458 - recall: 0.5604 - auc: 0.9056 - val_loss: 0.0030 - val_tp: 58.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 23.0000 - val_accuracy: 0.9993 - val_precision: 0.8923 - val_recall: 0.7160 - val_auc: 0.9565
Epoch 13/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0046 - tp: 188.0000 - fp: 35.0000 - tn: 181918.0000 - fn: 135.0000 - accuracy: 0.9991 - precision: 0.8430 - recall: 0.5820 - auc: 0.9229 - val_loss: 0.0030 - val_tp: 61.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 20.0000 - val_accuracy: 0.9994 - val_precision: 0.8971 - val_recall: 0.7531 - val_auc: 0.9565
Epoch 14/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0048 - tp: 188.0000 - fp: 37.0000 - tn: 181916.0000 - fn: 135.0000 - accuracy: 0.9991 - precision: 0.8356 - recall: 0.5820 - auc: 0.9119 - val_loss: 0.0030 - val_tp: 59.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 22.0000 - val_accuracy: 0.9994 - val_precision: 0.8939 - val_recall: 0.7284 - val_auc: 0.9565
Epoch 15/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0046 - tp: 179.0000 - fp: 30.0000 - tn: 181923.0000 - fn: 144.0000 - accuracy: 0.9990 - precision: 0.8565 - recall: 0.5542 - auc: 0.9261 - val_loss: 0.0029 - val_tp: 62.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 19.0000 - val_accuracy: 0.9994 - val_precision: 0.8986 - val_recall: 0.7654 - val_auc: 0.9565
Epoch 16/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0046 - tp: 176.0000 - fp: 41.0000 - tn: 181912.0000 - fn: 147.0000 - accuracy: 0.9990 - precision: 0.8111 - recall: 0.5449 - auc: 0.9260 - val_loss: 0.0029 - val_tp: 62.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 19.0000 - val_accuracy: 0.9994 - val_precision: 0.8986 - val_recall: 0.7654 - val_auc: 0.9565
Epoch 17/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0042 - tp: 194.0000 - fp: 35.0000 - tn: 181918.0000 - fn: 129.0000 - accuracy: 0.9991 - precision: 0.8472 - recall: 0.6006 - auc: 0.9292 - val_loss: 0.0029 - val_tp: 63.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 18.0000 - val_accuracy: 0.9995 - val_precision: 0.9000 - val_recall: 0.7778 - val_auc: 0.9565
Epoch 18/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0044 - tp: 201.0000 - fp: 34.0000 - tn: 181919.0000 - fn: 122.0000 - accuracy: 0.9991 - precision: 0.8553 - recall: 0.6223 - auc: 0.9199 - val_loss: 0.0029 - val_tp: 62.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 19.0000 - val_accuracy: 0.9994 - val_precision: 0.8986 - val_recall: 0.7654 - val_auc: 0.9565
Epoch 19/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0044 - tp: 189.0000 - fp: 33.0000 - tn: 181920.0000 - fn: 134.0000 - accuracy: 0.9991 - precision: 0.8514 - recall: 0.5851 - auc: 0.9291 - val_loss: 0.0029 - val_tp: 63.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 18.0000 - val_accuracy: 0.9995 - val_precision: 0.9000 - val_recall: 0.7778 - val_auc: 0.9565
Epoch 20/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0043 - tp: 184.0000 - fp: 30.0000 - tn: 181923.0000 - fn: 139.0000 - accuracy: 0.9991 - precision: 0.8598 - recall: 0.5697 - auc: 0.9277 - val_loss: 0.0029 - val_tp: 64.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 17.0000 - val_accuracy: 0.9995 - val_precision: 0.9014 - val_recall: 0.7901 - val_auc: 0.9565
Epoch 21/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0043 - tp: 193.0000 - fp: 38.0000 - tn: 181915.0000 - fn: 130.0000 - accuracy: 0.9991 - precision: 0.8355 - recall: 0.5975 - auc: 0.9262 - val_loss: 0.0028 - val_tp: 63.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 18.0000 - val_accuracy: 0.9995 - val_precision: 0.9000 - val_recall: 0.7778 - val_auc: 0.9565
Epoch 22/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0041 - tp: 193.0000 - fp: 35.0000 - tn: 181918.0000 - fn: 130.0000 - accuracy: 0.9991 - precision: 0.8465 - recall: 0.5975 - auc: 0.9324 - val_loss: 0.0028 - val_tp: 64.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 17.0000 - val_accuracy: 0.9995 - val_precision: 0.9014 - val_recall: 0.7901 - val_auc: 0.9565
Epoch 23/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0044 - tp: 192.0000 - fp: 31.0000 - tn: 181922.0000 - fn: 131.0000 - accuracy: 0.9991 - precision: 0.8610 - recall: 0.5944 - auc: 0.9277 - val_loss: 0.0028 - val_tp: 64.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 17.0000 - val_accuracy: 0.9995 - val_precision: 0.9014 - val_recall: 0.7901 - val_auc: 0.9565
Epoch 24/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0042 - tp: 201.0000 - fp: 40.0000 - tn: 181913.0000 - fn: 122.0000 - accuracy: 0.9991 - precision: 0.8340 - recall: 0.6223 - auc: 0.9247 - val_loss: 0.0028 - val_tp: 63.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 18.0000 - val_accuracy: 0.9995 - val_precision: 0.9000 - val_recall: 0.7778 - val_auc: 0.9565
Epoch 25/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0040 - tp: 200.0000 - fp: 31.0000 - tn: 181922.0000 - fn: 123.0000 - accuracy: 0.9992 - precision: 0.8658 - recall: 0.6192 - auc: 0.9309 - val_loss: 0.0028 - val_tp: 59.0000 - val_fp: 3.0000 - val_tn: 45485.0000 - val_fn: 22.0000 - val_accuracy: 0.9995 - val_precision: 0.9516 - val_recall: 0.7284 - val_auc: 0.9565
Epoch 26/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0042 - tp: 191.0000 - fp: 32.0000 - tn: 181921.0000 - fn: 132.0000 - accuracy: 0.9991 - precision: 0.8565 - recall: 0.5913 - auc: 0.9261 - val_loss: 0.0028 - val_tp: 65.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 16.0000 - val_accuracy: 0.9995 - val_precision: 0.9028 - val_recall: 0.8025 - val_auc: 0.9565
Epoch 27/100
90/90 [==============================] - 1s 6ms/step - loss: 0.0041 - tp: 202.0000 - fp: 35.0000 - tn: 181918.0000 - fn: 121.0000 - accuracy: 0.9991 - precision: 0.8523 - recall: 0.6254 - auc: 0.9324 - val_loss: 0.0028 - val_tp: 64.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 17.0000 - val_accuracy: 0.9995 - val_precision: 0.9014 - val_recall: 0.7901 - val_auc: 0.9565
Epoch 28/100
90/90 [==============================] - ETA: 0s - loss: 0.0042 - tp: 204.0000 - fp: 35.0000 - tn: 181918.0000 - fn: 119.0000 - accuracy: 0.9992 - precision: 0.8536 - recall: 0.6316 - auc: 0.9308Restoring model weights from the end of the best epoch.
90/90 [==============================] - 1s 6ms/step - loss: 0.0042 - tp: 204.0000 - fp: 35.0000 - tn: 181918.0000 - fn: 119.0000 - accuracy: 0.9992 - precision: 0.8536 - recall: 0.6316 - auc: 0.9308 - val_loss: 0.0028 - val_tp: 63.0000 - val_fp: 7.0000 - val_tn: 45481.0000 - val_fn: 18.0000 - val_accuracy: 0.9995 - val_precision: 0.9000 - val_recall: 0.7778 - val_auc: 0.9565
Epoch 00028: early stopping
"""

#Check training history
#We produce plots of your model's accuracy and loss on the training and validation set.
# These are useful to check for overfitting.

def plot_metrics(history):
  metrics =  ['loss', 'auc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()


plot_metrics(baseline_history)

#That the validation curve generally performs better than the training curve.
# This is mainly caused by the fact that the dropout layer is not active when evaluating the model.

#Evaluate metrics
#You can use a confusion matrix to summarize the actual vs. predicted labels where the X axis is the predicted label and the Y axis is the actual label.

train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)

def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
  print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
  print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
  print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
  print('Total Fraudulent Transactions: ', np.sum(cm[1]))

#Evaluate your model on the test dataset and display the results for the metrics you created above.

baseline_results = model.evaluate(test_features, test_labels,
                                  batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(model.metrics_names, baseline_results):
  print(name, ': ', value)
print()

plot_cm(test_labels, test_predictions_baseline)

"""
loss :  0.0022254730574786663
tp :  72.0
fp :  9.0
tn :  56865.0
fn :  16.0
accuracy :  0.9995611310005188
precision :  0.8888888955116272
recall :  0.8181818127632141
auc :  0.954325258731842

Legitimate Transactions Detected (True Negatives):  56865
Legitimate Transactions Incorrectly Detected (False Positives):  9
Fraudulent Transactions Missed (False Negatives):  16
Fraudulent Transactions Detected (True Positives):  72
Total Fraudulent Transactions:  88
"""

#If the model had predicted everything perfectly, this would be a diagonal matrix where values off the main diagonal, indicating incorrect predictions, would be zero.
# In this case the matrix shows that you have relatively few false positives, meaning that there were relatively few legitimate transactions that were incorrectly flagged.
# However, you would likely want to have even fewer false negatives despite the cost of increasing the number of false positives.
# This trade off may be preferable because false negatives would allow fraudulent transactions to go through, whereas false positives may cause an email to be sent to a customer to ask them to verify their card activity.

#Now plot the ROC.
# This plot is useful because it shows, at a glance, the range of performance the model can reach just by tuning the output threshold.

def plot_roc(name, labels, predictions, **kwargs):
  fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.xlim([-0.5,20])
  plt.ylim([80,100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal')

plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
plt.legend(loc='lower right')

#<matplotlib.legend.Legend at 0x7f28900702b0>

#It looks like the precision is relatively high, but the recall and the area under the ROC curve (AUC) aren't as high as you might like.
# Classifiers often face challenges when trying to maximize both precision and recall, which is especially true when working with imbalanced datasets.
# It is important to consider the costs of different types of errors in the context of the problem you care about.
# In this example, a false negative (a fraudulent transaction is missed) may have a financial cost, while a false positive (a transaction is incorrectly flagged as fraudulent) may decrease user happiness.

#Class weights

#Calculate class weights
#The goal is to identify fraudulent transactions, but you don't have very many of those positive samples to work with, so you would want to have the classifier heavily weight the few examples that are available.
# You can do this by passing Keras weights for each class through a parameter.
# These will cause the model to "pay more attention" to examples from an under-represented class.

# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg)*(total)/2.0
weight_for_1 = (1 / pos)*(total)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

#Weight for class 0: 0.50
#Weight for class 1: 289.44

#Train a model with class weights

weighted_model = make_model()
weighted_model.load_weights(initial_weights)

weighted_history = weighted_model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks = [early_stopping],
    validation_data=(val_features, val_labels),
    # The class weights go here
    class_weight=class_weight)

"""
Epoch 1/100
90/90 [==============================] - 1s 13ms/step - loss: 1.8005 - tp: 161.0000 - fp: 1170.0000 - tn: 237657.0000 - fn: 250.0000 - accuracy: 0.9941 - precision: 0.1210 - recall: 0.3917 - auc: 0.8086 - val_loss: 0.0214 - val_tp: 57.0000 - val_fp: 180.0000 - val_tn: 45308.0000 - val_fn: 24.0000 - val_accuracy: 0.9955 - val_precision: 0.2405 - val_recall: 0.7037 - val_auc: 0.9622
Epoch 2/100
90/90 [==============================] - 1s 6ms/step - loss: 0.7343 - tp: 201.0000 - fp: 1841.0000 - tn: 180112.0000 - fn: 122.0000 - accuracy: 0.9892 - precision: 0.0984 - recall: 0.6223 - auc: 0.8987 - val_loss: 0.0294 - val_tp: 67.0000 - val_fp: 258.0000 - val_tn: 45230.0000 - val_fn: 14.0000 - val_accuracy: 0.9940 - val_precision: 0.2062 - val_recall: 0.8272 - val_auc: 0.9694
Epoch 3/100
90/90 [==============================] - 1s 6ms/step - loss: 0.4879 - tp: 239.0000 - fp: 2753.0000 - tn: 179200.0000 - fn: 84.0000 - accuracy: 0.9844 - precision: 0.0799 - recall: 0.7399 - auc: 0.9322 - val_loss: 0.0388 - val_tp: 69.0000 - val_fp: 354.0000 - val_tn: 45134.0000 - val_fn: 12.0000 - val_accuracy: 0.9920 - val_precision: 0.1631 - val_recall: 0.8519 - val_auc: 0.9740
Epoch 4/100
90/90 [==============================] - 1s 6ms/step - loss: 0.4625 - tp: 247.0000 - fp: 3728.0000 - tn: 178225.0000 - fn: 76.0000 - accuracy: 0.9791 - precision: 0.0621 - recall: 0.7647 - auc: 0.9211 - val_loss: 0.0476 - val_tp: 71.0000 - val_fp: 440.0000 - val_tn: 45048.0000 - val_fn: 10.0000 - val_accuracy: 0.9901 - val_precision: 0.1389 - val_recall: 0.8765 - val_auc: 0.9739
Epoch 5/100
90/90 [==============================] - 1s 6ms/step - loss: 0.4086 - tp: 265.0000 - fp: 4625.0000 - tn: 177328.0000 - fn: 58.0000 - accuracy: 0.9743 - precision: 0.0542 - recall: 0.8204 - auc: 0.9323 - val_loss: 0.0548 - val_tp: 72.0000 - val_fp: 540.0000 - val_tn: 44948.0000 - val_fn: 9.0000 - val_accuracy: 0.9880 - val_precision: 0.1176 - val_recall: 0.8889 - val_auc: 0.9737
Epoch 6/100
90/90 [==============================] - 1s 6ms/step - loss: 0.3618 - tp: 265.0000 - fp: 5593.0000 - tn: 176360.0000 - fn: 58.0000 - accuracy: 0.9690 - precision: 0.0452 - recall: 0.8204 - auc: 0.9444 - val_loss: 0.0654 - val_tp: 74.0000 - val_fp: 672.0000 - val_tn: 44816.0000 - val_fn: 7.0000 - val_accuracy: 0.9851 - val_precision: 0.0992 - val_recall: 0.9136 - val_auc: 0.9728
Epoch 7/100
90/90 [==============================] - 1s 6ms/step - loss: 0.3688 - tp: 268.0000 - fp: 6298.0000 - tn: 175655.0000 - fn: 55.0000 - accuracy: 0.9651 - precision: 0.0408 - recall: 0.8297 - auc: 0.9393 - val_loss: 0.0718 - val_tp: 74.0000 - val_fp: 791.0000 - val_tn: 44697.0000 - val_fn: 7.0000 - val_accuracy: 0.9825 - val_precision: 0.0855 - val_recall: 0.9136 - val_auc: 0.9732
Epoch 8/100
90/90 [==============================] - 1s 6ms/step - loss: 0.3438 - tp: 267.0000 - fp: 6826.0000 - tn: 175127.0000 - fn: 56.0000 - accuracy: 0.9622 - precision: 0.0376 - recall: 0.8266 - auc: 0.9455 - val_loss: 0.0788 - val_tp: 74.0000 - val_fp: 912.0000 - val_tn: 44576.0000 - val_fn: 7.0000 - val_accuracy: 0.9798 - val_precision: 0.0751 - val_recall: 0.9136 - val_auc: 0.9710
Epoch 9/100
90/90 [==============================] - 1s 6ms/step - loss: 0.3148 - tp: 279.0000 - fp: 7355.0000 - tn: 174598.0000 - fn: 44.0000 - accuracy: 0.9594 - precision: 0.0365 - recall: 0.8638 - auc: 0.9472 - val_loss: 0.0816 - val_tp: 74.0000 - val_fp: 955.0000 - val_tn: 44533.0000 - val_fn: 7.0000 - val_accuracy: 0.9789 - val_precision: 0.0719 - val_recall: 0.9136 - val_auc: 0.9710
Epoch 10/100
90/90 [==============================] - 1s 6ms/step - loss: 0.2727 - tp: 285.0000 - fp: 7495.0000 - tn: 174458.0000 - fn: 38.0000 - accuracy: 0.9587 - precision: 0.0366 - recall: 0.8824 - auc: 0.9583 - val_loss: 0.0838 - val_tp: 74.0000 - val_fp: 982.0000 - val_tn: 44506.0000 - val_fn: 7.0000 - val_accuracy: 0.9783 - val_precision: 0.0701 - val_recall: 0.9136 - val_auc: 0.9693
Epoch 11/100
90/90 [==============================] - 1s 6ms/step - loss: 0.3058 - tp: 278.0000 - fp: 7826.0000 - tn: 174127.0000 - fn: 45.0000 - accuracy: 0.9568 - precision: 0.0343 - recall: 0.8607 - auc: 0.9521 - val_loss: 0.0865 - val_tp: 74.0000 - val_fp: 998.0000 - val_tn: 44490.0000 - val_fn: 7.0000 - val_accuracy: 0.9779 - val_precision: 0.0690 - val_recall: 0.9136 - val_auc: 0.9694
Epoch 12/100
90/90 [==============================] - 1s 6ms/step - loss: 0.2583 - tp: 284.0000 - fp: 7745.0000 - tn: 174208.0000 - fn: 39.0000 - accuracy: 0.9573 - precision: 0.0354 - recall: 0.8793 - auc: 0.9640 - val_loss: 0.0881 - val_tp: 74.0000 - val_fp: 1023.0000 - val_tn: 44465.0000 - val_fn: 7.0000 - val_accuracy: 0.9774 - val_precision: 0.0675 - val_recall: 0.9136 - val_auc: 0.9688
Epoch 13/100
89/90 [============================>.] - ETA: 0s - loss: 0.2765 - tp: 280.0000 - fp: 8050.0000 - tn: 173899.0000 - fn: 43.0000 - accuracy: 0.9556 - precision: 0.0336 - recall: 0.8669 - auc: 0.9577Restoring model weights from the end of the best epoch.
90/90 [==============================] - 1s 6ms/step - loss: 0.2765 - tp: 280.0000 - fp: 8050.0000 - tn: 173903.0000 - fn: 43.0000 - accuracy: 0.9556 - precision: 0.0336 - recall: 0.8669 - auc: 0.9577 - val_loss: 0.0895 - val_tp: 74.0000 - val_fp: 1025.0000 - val_tn: 44463.0000 - val_fn: 7.0000 - val_accuracy: 0.9774 - val_precision: 0.0673 - val_recall: 0.9136 - val_auc: 0.9667
Epoch 00013: early stopping"""

#Check training history
plot_metrics(weighted_history)

#Evaluate metrics
rain_predictions_weighted = weighted_model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_weighted = weighted_model.predict(test_features, batch_size=BATCH_SIZE)

weighted_results = weighted_model.evaluate(test_features, test_labels,
                                           batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(weighted_model.metrics_names, weighted_results):
  print(name, ': ', value)
print()

plot_cm(test_labels, test_predictions_weighted)

"""
loss :  0.03673461452126503
tp :  77.0
fp :  408.0
tn :  56466.0
fn :  11.0
accuracy :  0.992644190788269
precision :  0.1587628871202469
recall :  0.875
auc :  0.9694361686706543

Legitimate Transactions Detected (True Negatives):  56466
Legitimate Transactions Incorrectly Detected (False Positives):  408
Fraudulent Transactions Missed (False Negatives):  11
Fraudulent Transactions Detected (True Positives):  77
Total Fraudulent Transactions:  88
"""

#Plot the ROC
plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')

plot_roc("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])
plot_roc("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')


plt.legend(loc='lower right')

#<matplotlib.legend.Legend at 0x7f28441b1940>

#Oversampling
#Oversample the minority class
#A related approach would be to resample the dataset by oversampling the minority class.

pos_features = train_features[bool_train_labels]
neg_features = train_features[~bool_train_labels]

pos_labels = train_labels[bool_train_labels]
neg_labels = train_labels[~bool_train_labels]

#Using NumPy
#You can balance the dataset manually by choosing the right number of random indices from the positive examples:

ids = np.arange(len(pos_features))
choices = np.random.choice(ids, len(neg_features))

res_pos_features = pos_features[choices]
res_pos_labels = pos_labels[choices]

res_pos_features.shape

#(181953, 29)

resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)

order = np.arange(len(resampled_labels))
np.random.shuffle(order)
resampled_features = resampled_features[order]
resampled_labels = resampled_labels[order]

resampled_features.shape

#(363906, 29)

#Using tf.data
#If you're using tf.data the easiest way to produce balanced examples is to start with a positive and a negative dataset, and merge them.
# See the tf.data guide for more examples.

BUFFER_SIZE = 100000

def make_ds(features, labels):
  ds = tf.data.Dataset.from_tensor_slices((features, labels))#.cache()
  ds = ds.shuffle(BUFFER_SIZE).repeat()
  return ds

pos_ds = make_ds(pos_features, pos_labels)
neg_ds = make_ds(neg_features, neg_labels)

#Each dataset provides (feature, label) pairs:

for features, label in pos_ds.take(1):
  print("Features:\n", features.numpy())
  print()
  print("Label: ", label.numpy())

#Features:
# [-1.04278205e+00  2.00972213e+00 -2.59346703e+00  3.96162248e+00
# -2.24930440e+00 -9.40602521e-01 -4.70147821e+00  1.39978525e+00
# -4.06847795e+00 -5.00000000e+00  3.76549570e+00 -5.00000000e+00
#  1.50316118e+00 -5.00000000e+00  8.64022198e-01 -5.00000000e+00
# -5.00000000e+00 -5.00000000e+00  2.96298131e+00  1.42652399e+00
#  2.00568311e+00  1.14619550e+00 -2.30721370e-01  1.84304928e-03
# -7.47702385e-02  8.03987830e-01  3.26627629e+00  1.67797758e+00
# -4.50955643e-01]

#Label:  1


#Merge the two together using experimental.sample_from_datasets:

resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)

for features, label in resampled_ds.take(1):
  print(label.numpy().mean())

#0.48828125


#To use this dataset, we need the number of steps per epoch.

#The definition of "epoch" in this case is less clear.
# Say it's the number of batches required to see each negative example once:

resampled_steps_per_epoch = np.ceil(2.0*neg/BATCH_SIZE)
resampled_steps_per_epoch

#278.0

#Train on the oversampled data
#Now try training the model with the resampled data set instead of using class weights to see how these methods compare.

#Note: Because the data was balanced by replicating the positive examples, the total dataset size is larger, and each epoch runs for more training steps.
resampled_model = make_model()
resampled_model.load_weights(initial_weights)

# Reset the bias to zero, since this dataset is balanced.
output_layer = resampled_model.layers[-1]
output_layer.bias.assign([0])

val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).cache()
val_ds = val_ds.batch(BATCH_SIZE).prefetch(2)

resampled_history = resampled_model.fit(
    resampled_ds,
    epochs=EPOCHS,
    steps_per_epoch=resampled_steps_per_epoch,
    callbacks = [early_stopping],
    validation_data=val_ds)
"""
Epoch 1/100
278/278 [==============================] - 6s 22ms/step - loss: 0.4534 - tp: 252105.0000 - fp: 95834.0000 - tn: 246188.0000 - fn: 32179.0000 - accuracy: 0.7956 - precision: 0.7246 - recall: 0.8868 - auc: 0.9172 - val_loss: 0.2788 - val_tp: 74.0000 - val_fp: 1854.0000 - val_tn: 43634.0000 - val_fn: 7.0000 - val_accuracy: 0.9592 - val_precision: 0.0384 - val_recall: 0.9136 - val_auc: 0.9650
Epoch 2/100
278/278 [==============================] - 6s 20ms/step - loss: 0.2198 - tp: 261578.0000 - fp: 22578.0000 - tn: 261788.0000 - fn: 23400.0000 - accuracy: 0.9192 - precision: 0.9205 - recall: 0.9179 - auc: 0.9707 - val_loss: 0.1562 - val_tp: 74.0000 - val_fp: 1178.0000 - val_tn: 44310.0000 - val_fn: 7.0000 - val_accuracy: 0.9740 - val_precision: 0.0591 - val_recall: 0.9136 - val_auc: 0.9682
Epoch 3/100
278/278 [==============================] - 5s 20ms/step - loss: 0.1751 - tp: 263453.0000 - fp: 14271.0000 - tn: 270033.0000 - fn: 21587.0000 - accuracy: 0.9370 - precision: 0.9486 - recall: 0.9243 - auc: 0.9807 - val_loss: 0.1160 - val_tp: 74.0000 - val_fp: 1105.0000 - val_tn: 44383.0000 - val_fn: 7.0000 - val_accuracy: 0.9756 - val_precision: 0.0628 - val_recall: 0.9136 - val_auc: 0.9724
Epoch 4/100
278/278 [==============================] - 6s 20ms/step - loss: 0.1539 - tp: 264978.0000 - fp: 11989.0000 - tn: 273054.0000 - fn: 19323.0000 - accuracy: 0.9450 - precision: 0.9567 - recall: 0.9320 - auc: 0.9854 - val_loss: 0.0954 - val_tp: 74.0000 - val_fp: 1009.0000 - val_tn: 44479.0000 - val_fn: 7.0000 - val_accuracy: 0.9777 - val_precision: 0.0683 - val_recall: 0.9136 - val_auc: 0.9755
Epoch 5/100
278/278 [==============================] - 6s 20ms/step - loss: 0.1399 - tp: 267097.0000 - fp: 10771.0000 - tn: 273591.0000 - fn: 17885.0000 - accuracy: 0.9497 - precision: 0.9612 - recall: 0.9372 - auc: 0.9881 - val_loss: 0.0851 - val_tp: 74.0000 - val_fp: 986.0000 - val_tn: 44502.0000 - val_fn: 7.0000 - val_accuracy: 0.9782 - val_precision: 0.0698 - val_recall: 0.9136 - val_auc: 0.9772
Epoch 6/100
278/278 [==============================] - 6s 20ms/step - loss: 0.1300 - tp: 267754.0000 - fp: 10025.0000 - tn: 274618.0000 - fn: 16947.0000 - accuracy: 0.9526 - precision: 0.9639 - recall: 0.9405 - auc: 0.9901 - val_loss: 0.0760 - val_tp: 74.0000 - val_fp: 910.0000 - val_tn: 44578.0000 - val_fn: 7.0000 - val_accuracy: 0.9799 - val_precision: 0.0752 - val_recall: 0.9136 - val_auc: 0.9785
Epoch 7/100
278/278 [==============================] - 6s 20ms/step - loss: 0.1217 - tp: 268955.0000 - fp: 9284.0000 - tn: 274931.0000 - fn: 16174.0000 - accuracy: 0.9553 - precision: 0.9666 - recall: 0.9433 - auc: 0.9914 - val_loss: 0.0678 - val_tp: 74.0000 - val_fp: 819.0000 - val_tn: 44669.0000 - val_fn: 7.0000 - val_accuracy: 0.9819 - val_precision: 0.0829 - val_recall: 0.9136 - val_auc: 0.9759
Epoch 8/100
278/278 [==============================] - 6s 21ms/step - loss: 0.1154 - tp: 269154.0000 - fp: 8726.0000 - tn: 275782.0000 - fn: 15682.0000 - accuracy: 0.9571 - precision: 0.9686 - recall: 0.9449 - auc: 0.9924 - val_loss: 0.0645 - val_tp: 74.0000 - val_fp: 810.0000 - val_tn: 44678.0000 - val_fn: 7.0000 - val_accuracy: 0.9821 - val_precision: 0.0837 - val_recall: 0.9136 - val_auc: 0.9757
Epoch 9/100
278/278 [==============================] - 6s 21ms/step - loss: 0.1103 - tp: 269389.0000 - fp: 8508.0000 - tn: 276383.0000 - fn: 15064.0000 - accuracy: 0.9586 - precision: 0.9694 - recall: 0.9470 - auc: 0.9932 - val_loss: 0.0585 - val_tp: 74.0000 - val_fp: 720.0000 - val_tn: 44768.0000 - val_fn: 7.0000 - val_accuracy: 0.9840 - val_precision: 0.0932 - val_recall: 0.9136 - val_auc: 0.9763
Epoch 10/100
278/278 [==============================] - 5s 20ms/step - loss: 0.1051 - tp: 269899.0000 - fp: 8100.0000 - tn: 276730.0000 - fn: 14615.0000 - accuracy: 0.9601 - precision: 0.9709 - recall: 0.9486 - auc: 0.9939 - val_loss: 0.0542 - val_tp: 74.0000 - val_fp: 665.0000 - val_tn: 44823.0000 - val_fn: 7.0000 - val_accuracy: 0.9853 - val_precision: 0.1001 - val_recall: 0.9136 - val_auc: 0.9762
Epoch 11/100
278/278 [==============================] - 6s 20ms/step - loss: 0.1014 - tp: 269907.0000 - fp: 7844.0000 - tn: 277427.0000 - fn: 14166.0000 - accuracy: 0.9613 - precision: 0.9718 - recall: 0.9501 - auc: 0.9945 - val_loss: 0.0490 - val_tp: 73.0000 - val_fp: 591.0000 - val_tn: 44897.0000 - val_fn: 8.0000 - val_accuracy: 0.9869 - val_precision: 0.1099 - val_recall: 0.9012 - val_auc: 0.9764
Epoch 12/100
278/278 [==============================] - 6s 20ms/step - loss: 0.0976 - tp: 271444.0000 - fp: 7575.0000 - tn: 276328.0000 - fn: 13997.0000 - accuracy: 0.9621 - precision: 0.9729 - recall: 0.9510 - auc: 0.9948 - val_loss: 0.0470 - val_tp: 73.0000 - val_fp: 606.0000 - val_tn: 44882.0000 - val_fn: 8.0000 - val_accuracy: 0.9865 - val_precision: 0.1075 - val_recall: 0.9012 - val_auc: 0.9763
Epoch 13/100
278/278 [==============================] - 6s 20ms/step - loss: 0.0951 - tp: 270934.0000 - fp: 7432.0000 - tn: 277298.0000 - fn: 13680.0000 - accuracy: 0.9629 - precision: 0.9733 - recall: 0.9519 - auc: 0.9950 - val_loss: 0.0447 - val_tp: 73.0000 - val_fp: 584.0000 - val_tn: 44904.0000 - val_fn: 8.0000 - val_accuracy: 0.9870 - val_precision: 0.1111 - val_recall: 0.9012 - val_auc: 0.9768
Epoch 14/100
278/278 [==============================] - 5s 20ms/step - loss: 0.0926 - tp: 271538.0000 - fp: 7360.0000 - tn: 277032.0000 - fn: 13414.0000 - accuracy: 0.9635 - precision: 0.9736 - recall: 0.9529 - auc: 0.9952 - val_loss: 0.0428 - val_tp: 73.0000 - val_fp: 563.0000 - val_tn: 44925.0000 - val_fn: 8.0000 - val_accuracy: 0.9875 - val_precision: 0.1148 - val_recall: 0.9012 - val_auc: 0.9721
Epoch 15/100
278/278 [==============================] - 6s 20ms/step - loss: 0.0912 - tp: 271288.0000 - fp: 7125.0000 - tn: 277959.0000 - fn: 12972.0000 - accuracy: 0.9647 - precision: 0.9744 - recall: 0.9544 - auc: 0.9953 - val_loss: 0.0387 - val_tp: 73.0000 - val_fp: 507.0000 - val_tn: 44981.0000 - val_fn: 8.0000 - val_accuracy: 0.9887 - val_precision: 0.1259 - val_recall: 0.9012 - val_auc: 0.9716
Epoch 16/100
278/278 [==============================] - ETA: 0s - loss: 0.0892 - tp: 272354.0000 - fp: 7032.0000 - tn: 277250.0000 - fn: 12708.0000 - accuracy: 0.9653 - precision: 0.9748 - recall: 0.9554 - auc: 0.9955Restoring model weights from the end of the best epoch.
278/278 [==============================] - 5s 20ms/step - loss: 0.0892 - tp: 272354.0000 - fp: 7032.0000 - tn: 277250.0000 - fn: 12708.0000 - accuracy: 0.9653 - precision: 0.9748 - recall: 0.9554 - auc: 0.9955 - val_loss: 0.0395 - val_tp: 73.0000 - val_fp: 513.0000 - val_tn: 44975.0000 - val_fn: 8.0000 - val_accuracy: 0.9886 - val_precision: 0.1246 - val_recall: 0.9012 - val_auc: 0.9671
Epoch 00016: early stopping
"""

#If the training process were considering the whole dataset on each gradient update, this oversampling would be basically identical to the class weighting.

#But when training the model batch-wise, as you did here, the oversampled data provides a smoother gradient signal:
#Instead of each positive example being shown in one batch with a large weight, they're shown in many different batches each time with a small weight.
#This smoother gradient signal makes it easier to train the model.

#Re-train
#Because training is easier on the balanced data, the above training procedure may overfit quickly.

#So break up the epochs to give the callbacks.EarlyStopping finer control over when to stop training.

resampled_model = make_model()
resampled_model.load_weights(initial_weights)

# Reset the bias to zero, since this dataset is balanced.
output_layer = resampled_model.layers[-1]
output_layer.bias.assign([0])

resampled_history = resampled_model.fit(
    resampled_ds,
    # These are not real epochs
    steps_per_epoch = 20,
    epochs=10*EPOCHS,
    callbacks = [early_stopping],
    validation_data=(val_ds))

"""
Epoch 1/1000
20/20 [==============================] - 1s 50ms/step - loss: 1.0108 - tp: 14123.0000 - fp: 14257.0000 - tn: 51670.0000 - fn: 6479.0000 - accuracy: 0.7604 - precision: 0.4976 - recall: 0.6855 - auc: 0.8545 - val_loss: 1.0800 - val_tp: 80.0000 - val_fp: 34164.0000 - val_tn: 11324.0000 - val_fn: 1.0000 - val_accuracy: 0.2503 - val_precision: 0.0023 - val_recall: 0.9877 - val_auc: 0.8998
Epoch 2/1000
20/20 [==============================] - 0s 22ms/step - loss: 0.7262 - tp: 17347.0000 - fp: 12752.0000 - tn: 7694.0000 - fn: 3167.0000 - accuracy: 0.6114 - precision: 0.5763 - recall: 0.8456 - auc: 0.7821 - val_loss: 0.9532 - val_tp: 80.0000 - val_fp: 30230.0000 - val_tn: 15258.0000 - val_fn: 1.0000 - val_accuracy: 0.3366 - val_precision: 0.0026 - val_recall: 0.9877 - val_auc: 0.9574
Epoch 3/1000
20/20 [==============================] - 0s 22ms/step - loss: 0.6069 - tp: 18120.0000 - fp: 11692.0000 - tn: 8925.0000 - fn: 2223.0000 - accuracy: 0.6603 - precision: 0.6078 - recall: 0.8907 - auc: 0.8539 - val_loss: 0.8262 - val_tp: 79.0000 - val_fp: 23976.0000 - val_tn: 21512.0000 - val_fn: 2.0000 - val_accuracy: 0.4738 - val_precision: 0.0033 - val_recall: 0.9753 - val_auc: 0.9634
Epoch 4/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.5269 - tp: 18480.0000 - fp: 10021.0000 - tn: 10465.0000 - fn: 1994.0000 - accuracy: 0.7067 - precision: 0.6484 - recall: 0.9026 - auc: 0.8870 - val_loss: 0.7167 - val_tp: 79.0000 - val_fp: 17807.0000 - val_tn: 27681.0000 - val_fn: 2.0000 - val_accuracy: 0.6092 - val_precision: 0.0044 - val_recall: 0.9753 - val_auc: 0.9650
Epoch 5/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.4707 - tp: 18546.0000 - fp: 8531.0000 - tn: 12003.0000 - fn: 1880.0000 - accuracy: 0.7458 - precision: 0.6849 - recall: 0.9080 - auc: 0.9057 - val_loss: 0.6275 - val_tp: 77.0000 - val_fp: 12701.0000 - val_tn: 32787.0000 - val_fn: 4.0000 - val_accuracy: 0.7212 - val_precision: 0.0060 - val_recall: 0.9506 - val_auc: 0.9650
Epoch 6/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.4264 - tp: 18661.0000 - fp: 7163.0000 - tn: 13263.0000 - fn: 1873.0000 - accuracy: 0.7794 - precision: 0.7226 - recall: 0.9088 - auc: 0.9184 - val_loss: 0.5551 - val_tp: 77.0000 - val_fp: 9099.0000 - val_tn: 36389.0000 - val_fn: 4.0000 - val_accuracy: 0.8002 - val_precision: 0.0084 - val_recall: 0.9506 - val_auc: 0.9642
Epoch 7/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.3892 - tp: 18629.0000 - fp: 6092.0000 - tn: 14399.0000 - fn: 1840.0000 - accuracy: 0.8063 - precision: 0.7536 - recall: 0.9101 - auc: 0.9299 - val_loss: 0.4951 - val_tp: 77.0000 - val_fp: 6377.0000 - val_tn: 39111.0000 - val_fn: 4.0000 - val_accuracy: 0.8600 - val_precision: 0.0119 - val_recall: 0.9506 - val_auc: 0.9638
Epoch 8/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.3606 - tp: 18708.0000 - fp: 5232.0000 - tn: 15205.0000 - fn: 1815.0000 - accuracy: 0.8280 - precision: 0.7815 - recall: 0.9116 - auc: 0.9364 - val_loss: 0.4461 - val_tp: 77.0000 - val_fp: 4719.0000 - val_tn: 40769.0000 - val_fn: 4.0000 - val_accuracy: 0.8964 - val_precision: 0.0161 - val_recall: 0.9506 - val_auc: 0.9637
Epoch 9/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.3349 - tp: 18808.0000 - fp: 4398.0000 - tn: 15901.0000 - fn: 1853.0000 - accuracy: 0.8474 - precision: 0.8105 - recall: 0.9103 - auc: 0.9432 - val_loss: 0.4067 - val_tp: 76.0000 - val_fp: 3711.0000 - val_tn: 41777.0000 - val_fn: 5.0000 - val_accuracy: 0.9185 - val_precision: 0.0201 - val_recall: 0.9383 - val_auc: 0.9639
Epoch 10/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.3178 - tp: 18674.0000 - fp: 3878.0000 - tn: 16602.0000 - fn: 1806.0000 - accuracy: 0.8612 - precision: 0.8280 - recall: 0.9118 - auc: 0.9471 - val_loss: 0.3730 - val_tp: 76.0000 - val_fp: 3081.0000 - val_tn: 42407.0000 - val_fn: 5.0000 - val_accuracy: 0.9323 - val_precision: 0.0241 - val_recall: 0.9383 - val_auc: 0.9641
Epoch 11/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.2997 - tp: 18850.0000 - fp: 3342.0000 - tn: 16964.0000 - fn: 1804.0000 - accuracy: 0.8744 - precision: 0.8494 - recall: 0.9127 - auc: 0.9519 - val_loss: 0.3452 - val_tp: 76.0000 - val_fp: 2653.0000 - val_tn: 42835.0000 - val_fn: 5.0000 - val_accuracy: 0.9417 - val_precision: 0.0278 - val_recall: 0.9383 - val_auc: 0.9647
Epoch 12/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.2876 - tp: 18758.0000 - fp: 3227.0000 - tn: 17182.0000 - fn: 1793.0000 - accuracy: 0.8774 - precision: 0.8532 - recall: 0.9128 - auc: 0.9550 - val_loss: 0.3210 - val_tp: 74.0000 - val_fp: 2334.0000 - val_tn: 43154.0000 - val_fn: 7.0000 - val_accuracy: 0.9486 - val_precision: 0.0307 - val_recall: 0.9136 - val_auc: 0.9647
Epoch 13/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.2790 - tp: 18549.0000 - fp: 2864.0000 - tn: 17752.0000 - fn: 1795.0000 - accuracy: 0.8863 - precision: 0.8662 - recall: 0.9118 - auc: 0.9570 - val_loss: 0.2990 - val_tp: 74.0000 - val_fp: 2072.0000 - val_tn: 43416.0000 - val_fn: 7.0000 - val_accuracy: 0.9544 - val_precision: 0.0345 - val_recall: 0.9136 - val_auc: 0.9648
Epoch 14/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.2658 - tp: 18828.0000 - fp: 2519.0000 - tn: 17819.0000 - fn: 1794.0000 - accuracy: 0.8947 - precision: 0.8820 - recall: 0.9130 - auc: 0.9595 - val_loss: 0.2806 - val_tp: 74.0000 - val_fp: 1893.0000 - val_tn: 43595.0000 - val_fn: 7.0000 - val_accuracy: 0.9583 - val_precision: 0.0376 - val_recall: 0.9136 - val_auc: 0.9651
Epoch 15/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.2623 - tp: 18552.0000 - fp: 2466.0000 - tn: 18200.0000 - fn: 1742.0000 - accuracy: 0.8973 - precision: 0.8827 - recall: 0.9142 - auc: 0.9606 - val_loss: 0.2642 - val_tp: 74.0000 - val_fp: 1771.0000 - val_tn: 43717.0000 - val_fn: 7.0000 - val_accuracy: 0.9610 - val_precision: 0.0401 - val_recall: 0.9136 - val_auc: 0.9653
Epoch 16/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.2488 - tp: 18611.0000 - fp: 2136.0000 - tn: 18446.0000 - fn: 1767.0000 - accuracy: 0.9047 - precision: 0.8970 - recall: 0.9133 - auc: 0.9634 - val_loss: 0.2497 - val_tp: 74.0000 - val_fp: 1643.0000 - val_tn: 43845.0000 - val_fn: 7.0000 - val_accuracy: 0.9638 - val_precision: 0.0431 - val_recall: 0.9136 - val_auc: 0.9654
Epoch 17/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.2422 - tp: 18633.0000 - fp: 2027.0000 - tn: 18560.0000 - fn: 1740.0000 - accuracy: 0.9080 - precision: 0.9019 - recall: 0.9146 - auc: 0.9654 - val_loss: 0.2361 - val_tp: 74.0000 - val_fp: 1532.0000 - val_tn: 43956.0000 - val_fn: 7.0000 - val_accuracy: 0.9662 - val_precision: 0.0461 - val_recall: 0.9136 - val_auc: 0.9657
Epoch 18/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.2345 - tp: 18869.0000 - fp: 1824.0000 - tn: 18545.0000 - fn: 1722.0000 - accuracy: 0.9134 - precision: 0.9119 - recall: 0.9164 - auc: 0.9670 - val_loss: 0.2243 - val_tp: 74.0000 - val_fp: 1444.0000 - val_tn: 44044.0000 - val_fn: 7.0000 - val_accuracy: 0.9682 - val_precision: 0.0487 - val_recall: 0.9136 - val_auc: 0.9663
Epoch 19/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.2288 - tp: 18753.0000 - fp: 1687.0000 - tn: 18785.0000 - fn: 1735.0000 - accuracy: 0.9165 - precision: 0.9175 - recall: 0.9153 - auc: 0.9685 - val_loss: 0.2137 - val_tp: 74.0000 - val_fp: 1377.0000 - val_tn: 44111.0000 - val_fn: 7.0000 - val_accuracy: 0.9696 - val_precision: 0.0510 - val_recall: 0.9136 - val_auc: 0.9667
Epoch 20/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.2215 - tp: 18648.0000 - fp: 1582.0000 - tn: 19036.0000 - fn: 1694.0000 - accuracy: 0.9200 - precision: 0.9218 - recall: 0.9167 - auc: 0.9702 - val_loss: 0.2048 - val_tp: 74.0000 - val_fp: 1332.0000 - val_tn: 44156.0000 - val_fn: 7.0000 - val_accuracy: 0.9706 - val_precision: 0.0526 - val_recall: 0.9136 - val_auc: 0.9669
Epoch 21/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.2169 - tp: 18958.0000 - fp: 1563.0000 - tn: 18772.0000 - fn: 1667.0000 - accuracy: 0.9211 - precision: 0.9238 - recall: 0.9192 - auc: 0.9710 - val_loss: 0.1971 - val_tp: 74.0000 - val_fp: 1301.0000 - val_tn: 44187.0000 - val_fn: 7.0000 - val_accuracy: 0.9713 - val_precision: 0.0538 - val_recall: 0.9136 - val_auc: 0.9670
Epoch 22/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.2144 - tp: 18767.0000 - fp: 1517.0000 - tn: 19016.0000 - fn: 1660.0000 - accuracy: 0.9224 - precision: 0.9252 - recall: 0.9187 - auc: 0.9720 - val_loss: 0.1898 - val_tp: 74.0000 - val_fp: 1273.0000 - val_tn: 44215.0000 - val_fn: 7.0000 - val_accuracy: 0.9719 - val_precision: 0.0549 - val_recall: 0.9136 - val_auc: 0.9672
Epoch 23/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.2084 - tp: 18713.0000 - fp: 1398.0000 - tn: 19167.0000 - fn: 1682.0000 - accuracy: 0.9248 - precision: 0.9305 - recall: 0.9175 - auc: 0.9731 - val_loss: 0.1828 - val_tp: 74.0000 - val_fp: 1245.0000 - val_tn: 44243.0000 - val_fn: 7.0000 - val_accuracy: 0.9725 - val_precision: 0.0561 - val_recall: 0.9136 - val_auc: 0.9677
Epoch 24/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.2073 - tp: 18815.0000 - fp: 1397.0000 - tn: 19055.0000 - fn: 1693.0000 - accuracy: 0.9246 - precision: 0.9309 - recall: 0.9174 - auc: 0.9731 - val_loss: 0.1761 - val_tp: 74.0000 - val_fp: 1211.0000 - val_tn: 44277.0000 - val_fn: 7.0000 - val_accuracy: 0.9733 - val_precision: 0.0576 - val_recall: 0.9136 - val_auc: 0.9677
Epoch 25/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.2011 - tp: 18806.0000 - fp: 1270.0000 - tn: 19233.0000 - fn: 1651.0000 - accuracy: 0.9287 - precision: 0.9367 - recall: 0.9193 - auc: 0.9745 - val_loss: 0.1705 - val_tp: 74.0000 - val_fp: 1202.0000 - val_tn: 44286.0000 - val_fn: 7.0000 - val_accuracy: 0.9735 - val_precision: 0.0580 - val_recall: 0.9136 - val_auc: 0.9683
Epoch 26/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1996 - tp: 18659.0000 - fp: 1294.0000 - tn: 19408.0000 - fn: 1599.0000 - accuracy: 0.9294 - precision: 0.9351 - recall: 0.9211 - auc: 0.9750 - val_loss: 0.1647 - val_tp: 74.0000 - val_fp: 1181.0000 - val_tn: 44307.0000 - val_fn: 7.0000 - val_accuracy: 0.9739 - val_precision: 0.0590 - val_recall: 0.9136 - val_auc: 0.9682
Epoch 27/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1961 - tp: 18839.0000 - fp: 1265.0000 - tn: 19259.0000 - fn: 1597.0000 - accuracy: 0.9301 - precision: 0.9371 - recall: 0.9219 - auc: 0.9761 - val_loss: 0.1597 - val_tp: 74.0000 - val_fp: 1168.0000 - val_tn: 44320.0000 - val_fn: 7.0000 - val_accuracy: 0.9742 - val_precision: 0.0596 - val_recall: 0.9136 - val_auc: 0.9684
Epoch 28/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1930 - tp: 18784.0000 - fp: 1181.0000 - tn: 19375.0000 - fn: 1620.0000 - accuracy: 0.9316 - precision: 0.9408 - recall: 0.9206 - auc: 0.9764 - val_loss: 0.1546 - val_tp: 74.0000 - val_fp: 1141.0000 - val_tn: 44347.0000 - val_fn: 7.0000 - val_accuracy: 0.9748 - val_precision: 0.0609 - val_recall: 0.9136 - val_auc: 0.9687
Epoch 29/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1903 - tp: 18902.0000 - fp: 1128.0000 - tn: 19297.0000 - fn: 1633.0000 - accuracy: 0.9326 - precision: 0.9437 - recall: 0.9205 - auc: 0.9769 - val_loss: 0.1506 - val_tp: 74.0000 - val_fp: 1137.0000 - val_tn: 44351.0000 - val_fn: 7.0000 - val_accuracy: 0.9749 - val_precision: 0.0611 - val_recall: 0.9136 - val_auc: 0.9689
Epoch 30/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1876 - tp: 18779.0000 - fp: 1106.0000 - tn: 19449.0000 - fn: 1626.0000 - accuracy: 0.9333 - precision: 0.9444 - recall: 0.9203 - auc: 0.9774 - val_loss: 0.1471 - val_tp: 74.0000 - val_fp: 1136.0000 - val_tn: 44352.0000 - val_fn: 7.0000 - val_accuracy: 0.9749 - val_precision: 0.0612 - val_recall: 0.9136 - val_auc: 0.9692
Epoch 31/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1825 - tp: 18887.0000 - fp: 1062.0000 - tn: 19424.0000 - fn: 1587.0000 - accuracy: 0.9353 - precision: 0.9468 - recall: 0.9225 - auc: 0.9789 - val_loss: 0.1440 - val_tp: 74.0000 - val_fp: 1144.0000 - val_tn: 44344.0000 - val_fn: 7.0000 - val_accuracy: 0.9747 - val_precision: 0.0608 - val_recall: 0.9136 - val_auc: 0.9695
Epoch 32/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1784 - tp: 18835.0000 - fp: 992.0000 - tn: 19555.0000 - fn: 1578.0000 - accuracy: 0.9373 - precision: 0.9500 - recall: 0.9227 - auc: 0.9799 - val_loss: 0.1405 - val_tp: 74.0000 - val_fp: 1143.0000 - val_tn: 44345.0000 - val_fn: 7.0000 - val_accuracy: 0.9748 - val_precision: 0.0608 - val_recall: 0.9136 - val_auc: 0.9700
Epoch 33/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1791 - tp: 18964.0000 - fp: 1044.0000 - tn: 19360.0000 - fn: 1592.0000 - accuracy: 0.9356 - precision: 0.9478 - recall: 0.9226 - auc: 0.9796 - val_loss: 0.1373 - val_tp: 74.0000 - val_fp: 1131.0000 - val_tn: 44357.0000 - val_fn: 7.0000 - val_accuracy: 0.9750 - val_precision: 0.0614 - val_recall: 0.9136 - val_auc: 0.9701
Epoch 34/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1773 - tp: 18962.0000 - fp: 1022.0000 - tn: 19433.0000 - fn: 1543.0000 - accuracy: 0.9374 - precision: 0.9489 - recall: 0.9248 - auc: 0.9803 - val_loss: 0.1343 - val_tp: 74.0000 - val_fp: 1131.0000 - val_tn: 44357.0000 - val_fn: 7.0000 - val_accuracy: 0.9750 - val_precision: 0.0614 - val_recall: 0.9136 - val_auc: 0.9702
Epoch 35/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1764 - tp: 18956.0000 - fp: 1021.0000 - tn: 19451.0000 - fn: 1532.0000 - accuracy: 0.9377 - precision: 0.9489 - recall: 0.9252 - auc: 0.9808 - val_loss: 0.1309 - val_tp: 74.0000 - val_fp: 1108.0000 - val_tn: 44380.0000 - val_fn: 7.0000 - val_accuracy: 0.9755 - val_precision: 0.0626 - val_recall: 0.9136 - val_auc: 0.9707
Epoch 36/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1754 - tp: 19032.0000 - fp: 1045.0000 - tn: 19306.0000 - fn: 1577.0000 - accuracy: 0.9360 - precision: 0.9480 - recall: 0.9235 - auc: 0.9807 - val_loss: 0.1284 - val_tp: 74.0000 - val_fp: 1114.0000 - val_tn: 44374.0000 - val_fn: 7.0000 - val_accuracy: 0.9754 - val_precision: 0.0623 - val_recall: 0.9136 - val_auc: 0.9712
Epoch 37/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1700 - tp: 18931.0000 - fp: 993.0000 - tn: 19518.0000 - fn: 1518.0000 - accuracy: 0.9387 - precision: 0.9502 - recall: 0.9258 - auc: 0.9818 - val_loss: 0.1257 - val_tp: 74.0000 - val_fp: 1109.0000 - val_tn: 44379.0000 - val_fn: 7.0000 - val_accuracy: 0.9755 - val_precision: 0.0626 - val_recall: 0.9136 - val_auc: 0.9712
Epoch 38/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1704 - tp: 19023.0000 - fp: 937.0000 - tn: 19536.0000 - fn: 1464.0000 - accuracy: 0.9414 - precision: 0.9531 - recall: 0.9285 - auc: 0.9821 - val_loss: 0.1234 - val_tp: 74.0000 - val_fp: 1101.0000 - val_tn: 44387.0000 - val_fn: 7.0000 - val_accuracy: 0.9757 - val_precision: 0.0630 - val_recall: 0.9136 - val_auc: 0.9719
Epoch 39/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1666 - tp: 19106.0000 - fp: 922.0000 - tn: 19381.0000 - fn: 1551.0000 - accuracy: 0.9396 - precision: 0.9540 - recall: 0.9249 - auc: 0.9824 - val_loss: 0.1219 - val_tp: 74.0000 - val_fp: 1110.0000 - val_tn: 44378.0000 - val_fn: 7.0000 - val_accuracy: 0.9755 - val_precision: 0.0625 - val_recall: 0.9136 - val_auc: 0.9720
Epoch 40/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1670 - tp: 19049.0000 - fp: 965.0000 - tn: 19472.0000 - fn: 1474.0000 - accuracy: 0.9405 - precision: 0.9518 - recall: 0.9282 - auc: 0.9828 - val_loss: 0.1192 - val_tp: 74.0000 - val_fp: 1089.0000 - val_tn: 44399.0000 - val_fn: 7.0000 - val_accuracy: 0.9759 - val_precision: 0.0636 - val_recall: 0.9136 - val_auc: 0.9724
Epoch 41/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1631 - tp: 19127.0000 - fp: 927.0000 - tn: 19441.0000 - fn: 1465.0000 - accuracy: 0.9416 - precision: 0.9538 - recall: 0.9289 - auc: 0.9833 - val_loss: 0.1173 - val_tp: 74.0000 - val_fp: 1084.0000 - val_tn: 44404.0000 - val_fn: 7.0000 - val_accuracy: 0.9761 - val_precision: 0.0639 - val_recall: 0.9136 - val_auc: 0.9728
Epoch 42/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1616 - tp: 19035.0000 - fp: 913.0000 - tn: 19573.0000 - fn: 1439.0000 - accuracy: 0.9426 - precision: 0.9542 - recall: 0.9297 - auc: 0.9836 - val_loss: 0.1157 - val_tp: 74.0000 - val_fp: 1088.0000 - val_tn: 44400.0000 - val_fn: 7.0000 - val_accuracy: 0.9760 - val_precision: 0.0637 - val_recall: 0.9136 - val_auc: 0.9728
Epoch 43/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1608 - tp: 18934.0000 - fp: 903.0000 - tn: 19655.0000 - fn: 1468.0000 - accuracy: 0.9421 - precision: 0.9545 - recall: 0.9280 - auc: 0.9836 - val_loss: 0.1137 - val_tp: 74.0000 - val_fp: 1074.0000 - val_tn: 44414.0000 - val_fn: 7.0000 - val_accuracy: 0.9763 - val_precision: 0.0645 - val_recall: 0.9136 - val_auc: 0.9732
Epoch 44/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1601 - tp: 18866.0000 - fp: 929.0000 - tn: 19700.0000 - fn: 1465.0000 - accuracy: 0.9416 - precision: 0.9531 - recall: 0.9279 - auc: 0.9842 - val_loss: 0.1114 - val_tp: 74.0000 - val_fp: 1058.0000 - val_tn: 44430.0000 - val_fn: 7.0000 - val_accuracy: 0.9766 - val_precision: 0.0654 - val_recall: 0.9136 - val_auc: 0.9736
Epoch 45/1000
20/20 [==============================] - 0s 25ms/step - loss: 0.1573 - tp: 19041.0000 - fp: 853.0000 - tn: 19677.0000 - fn: 1389.0000 - accuracy: 0.9453 - precision: 0.9571 - recall: 0.9320 - auc: 0.9846 - val_loss: 0.1099 - val_tp: 74.0000 - val_fp: 1058.0000 - val_tn: 44430.0000 - val_fn: 7.0000 - val_accuracy: 0.9766 - val_precision: 0.0654 - val_recall: 0.9136 - val_auc: 0.9734
Epoch 46/1000
20/20 [==============================] - 0s 25ms/step - loss: 0.1577 - tp: 19032.0000 - fp: 900.0000 - tn: 19618.0000 - fn: 1410.0000 - accuracy: 0.9436 - precision: 0.9548 - recall: 0.9310 - auc: 0.9844 - val_loss: 0.1083 - val_tp: 74.0000 - val_fp: 1058.0000 - val_tn: 44430.0000 - val_fn: 7.0000 - val_accuracy: 0.9766 - val_precision: 0.0654 - val_recall: 0.9136 - val_auc: 0.9739
Epoch 47/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1553 - tp: 19181.0000 - fp: 861.0000 - tn: 19495.0000 - fn: 1423.0000 - accuracy: 0.9442 - precision: 0.9570 - recall: 0.9309 - auc: 0.9851 - val_loss: 0.1069 - val_tp: 74.0000 - val_fp: 1054.0000 - val_tn: 44434.0000 - val_fn: 7.0000 - val_accuracy: 0.9767 - val_precision: 0.0656 - val_recall: 0.9136 - val_auc: 0.9741
Epoch 48/1000
20/20 [==============================] - 1s 26ms/step - loss: 0.1552 - tp: 18946.0000 - fp: 885.0000 - tn: 19719.0000 - fn: 1410.0000 - accuracy: 0.9440 - precision: 0.9554 - recall: 0.9307 - auc: 0.9854 - val_loss: 0.1048 - val_tp: 74.0000 - val_fp: 1039.0000 - val_tn: 44449.0000 - val_fn: 7.0000 - val_accuracy: 0.9770 - val_precision: 0.0665 - val_recall: 0.9136 - val_auc: 0.9744
Epoch 49/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1537 - tp: 18943.0000 - fp: 829.0000 - tn: 19763.0000 - fn: 1425.0000 - accuracy: 0.9450 - precision: 0.9581 - recall: 0.9300 - auc: 0.9852 - val_loss: 0.1036 - val_tp: 74.0000 - val_fp: 1039.0000 - val_tn: 44449.0000 - val_fn: 7.0000 - val_accuracy: 0.9770 - val_precision: 0.0665 - val_recall: 0.9136 - val_auc: 0.9747
Epoch 50/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1564 - tp: 19158.0000 - fp: 872.0000 - tn: 19513.0000 - fn: 1417.0000 - accuracy: 0.9441 - precision: 0.9565 - recall: 0.9311 - auc: 0.9848 - val_loss: 0.1028 - val_tp: 74.0000 - val_fp: 1044.0000 - val_tn: 44444.0000 - val_fn: 7.0000 - val_accuracy: 0.9769 - val_precision: 0.0662 - val_recall: 0.9136 - val_auc: 0.9749
Epoch 51/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1498 - tp: 18961.0000 - fp: 839.0000 - tn: 19766.0000 - fn: 1394.0000 - accuracy: 0.9455 - precision: 0.9576 - recall: 0.9315 - auc: 0.9859 - val_loss: 0.1017 - val_tp: 74.0000 - val_fp: 1045.0000 - val_tn: 44443.0000 - val_fn: 7.0000 - val_accuracy: 0.9769 - val_precision: 0.0661 - val_recall: 0.9136 - val_auc: 0.9750
Epoch 52/1000
20/20 [==============================] - 0s 25ms/step - loss: 0.1508 - tp: 19039.0000 - fp: 839.0000 - tn: 19703.0000 - fn: 1379.0000 - accuracy: 0.9458 - precision: 0.9578 - recall: 0.9325 - auc: 0.9860 - val_loss: 0.1002 - val_tp: 74.0000 - val_fp: 1021.0000 - val_tn: 44467.0000 - val_fn: 7.0000 - val_accuracy: 0.9774 - val_precision: 0.0676 - val_recall: 0.9136 - val_auc: 0.9743
Epoch 53/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1458 - tp: 19052.0000 - fp: 782.0000 - tn: 19748.0000 - fn: 1378.0000 - accuracy: 0.9473 - precision: 0.9606 - recall: 0.9326 - auc: 0.9869 - val_loss: 0.0996 - val_tp: 74.0000 - val_fp: 1027.0000 - val_tn: 44461.0000 - val_fn: 7.0000 - val_accuracy: 0.9773 - val_precision: 0.0672 - val_recall: 0.9136 - val_auc: 0.9746
Epoch 54/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1489 - tp: 19260.0000 - fp: 820.0000 - tn: 19522.0000 - fn: 1358.0000 - accuracy: 0.9468 - precision: 0.9592 - recall: 0.9341 - auc: 0.9864 - val_loss: 0.0978 - val_tp: 74.0000 - val_fp: 1010.0000 - val_tn: 44478.0000 - val_fn: 7.0000 - val_accuracy: 0.9777 - val_precision: 0.0683 - val_recall: 0.9136 - val_auc: 0.9749
Epoch 55/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1474 - tp: 19032.0000 - fp: 840.0000 - tn: 19707.0000 - fn: 1381.0000 - accuracy: 0.9458 - precision: 0.9577 - recall: 0.9323 - auc: 0.9867 - val_loss: 0.0965 - val_tp: 74.0000 - val_fp: 1006.0000 - val_tn: 44482.0000 - val_fn: 7.0000 - val_accuracy: 0.9778 - val_precision: 0.0685 - val_recall: 0.9136 - val_auc: 0.9752
Epoch 56/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1467 - tp: 19218.0000 - fp: 828.0000 - tn: 19607.0000 - fn: 1307.0000 - accuracy: 0.9479 - precision: 0.9587 - recall: 0.9363 - auc: 0.9870 - val_loss: 0.0954 - val_tp: 74.0000 - val_fp: 999.0000 - val_tn: 44489.0000 - val_fn: 7.0000 - val_accuracy: 0.9779 - val_precision: 0.0690 - val_recall: 0.9136 - val_auc: 0.9755
Epoch 57/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1436 - tp: 19264.0000 - fp: 780.0000 - tn: 19573.0000 - fn: 1343.0000 - accuracy: 0.9482 - precision: 0.9611 - recall: 0.9348 - auc: 0.9871 - val_loss: 0.0950 - val_tp: 74.0000 - val_fp: 1014.0000 - val_tn: 44474.0000 - val_fn: 7.0000 - val_accuracy: 0.9776 - val_precision: 0.0680 - val_recall: 0.9136 - val_auc: 0.9757
Epoch 58/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1433 - tp: 19130.0000 - fp: 790.0000 - tn: 19721.0000 - fn: 1319.0000 - accuracy: 0.9485 - precision: 0.9603 - recall: 0.9355 - auc: 0.9875 - val_loss: 0.0940 - val_tp: 74.0000 - val_fp: 1008.0000 - val_tn: 44480.0000 - val_fn: 7.0000 - val_accuracy: 0.9777 - val_precision: 0.0684 - val_recall: 0.9136 - val_auc: 0.9760
Epoch 59/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1411 - tp: 19239.0000 - fp: 730.0000 - tn: 19659.0000 - fn: 1332.0000 - accuracy: 0.9497 - precision: 0.9634 - recall: 0.9352 - auc: 0.9878 - val_loss: 0.0940 - val_tp: 74.0000 - val_fp: 1028.0000 - val_tn: 44460.0000 - val_fn: 7.0000 - val_accuracy: 0.9773 - val_precision: 0.0672 - val_recall: 0.9136 - val_auc: 0.9760
Epoch 60/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1408 - tp: 19150.0000 - fp: 791.0000 - tn: 19733.0000 - fn: 1286.0000 - accuracy: 0.9493 - precision: 0.9603 - recall: 0.9371 - auc: 0.9879 - val_loss: 0.0931 - val_tp: 74.0000 - val_fp: 1028.0000 - val_tn: 44460.0000 - val_fn: 7.0000 - val_accuracy: 0.9773 - val_precision: 0.0672 - val_recall: 0.9136 - val_auc: 0.9763
Epoch 61/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1443 - tp: 19159.0000 - fp: 813.0000 - tn: 19686.0000 - fn: 1302.0000 - accuracy: 0.9484 - precision: 0.9593 - recall: 0.9364 - auc: 0.9873 - val_loss: 0.0916 - val_tp: 74.0000 - val_fp: 1007.0000 - val_tn: 44481.0000 - val_fn: 7.0000 - val_accuracy: 0.9777 - val_precision: 0.0685 - val_recall: 0.9136 - val_auc: 0.9753
Epoch 62/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1412 - tp: 19185.0000 - fp: 775.0000 - tn: 19723.0000 - fn: 1277.0000 - accuracy: 0.9499 - precision: 0.9612 - recall: 0.9376 - auc: 0.9879 - val_loss: 0.0903 - val_tp: 74.0000 - val_fp: 985.0000 - val_tn: 44503.0000 - val_fn: 7.0000 - val_accuracy: 0.9782 - val_precision: 0.0699 - val_recall: 0.9136 - val_auc: 0.9756
Epoch 63/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1409 - tp: 19256.0000 - fp: 776.0000 - tn: 19655.0000 - fn: 1273.0000 - accuracy: 0.9500 - precision: 0.9613 - recall: 0.9380 - auc: 0.9880 - val_loss: 0.0895 - val_tp: 74.0000 - val_fp: 986.0000 - val_tn: 44502.0000 - val_fn: 7.0000 - val_accuracy: 0.9782 - val_precision: 0.0698 - val_recall: 0.9136 - val_auc: 0.9758
Epoch 64/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1395 - tp: 19037.0000 - fp: 758.0000 - tn: 19848.0000 - fn: 1317.0000 - accuracy: 0.9493 - precision: 0.9617 - recall: 0.9353 - auc: 0.9881 - val_loss: 0.0887 - val_tp: 74.0000 - val_fp: 975.0000 - val_tn: 44513.0000 - val_fn: 7.0000 - val_accuracy: 0.9785 - val_precision: 0.0705 - val_recall: 0.9136 - val_auc: 0.9760
Epoch 65/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1405 - tp: 19236.0000 - fp: 766.0000 - tn: 19663.0000 - fn: 1295.0000 - accuracy: 0.9497 - precision: 0.9617 - recall: 0.9369 - auc: 0.9881 - val_loss: 0.0883 - val_tp: 74.0000 - val_fp: 985.0000 - val_tn: 44503.0000 - val_fn: 7.0000 - val_accuracy: 0.9782 - val_precision: 0.0699 - val_recall: 0.9136 - val_auc: 0.9762
Epoch 66/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1367 - tp: 19327.0000 - fp: 731.0000 - tn: 19629.0000 - fn: 1273.0000 - accuracy: 0.9511 - precision: 0.9636 - recall: 0.9382 - auc: 0.9886 - val_loss: 0.0877 - val_tp: 74.0000 - val_fp: 978.0000 - val_tn: 44510.0000 - val_fn: 7.0000 - val_accuracy: 0.9784 - val_precision: 0.0703 - val_recall: 0.9136 - val_auc: 0.9765
Epoch 67/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1377 - tp: 19210.0000 - fp: 741.0000 - tn: 19747.0000 - fn: 1262.0000 - accuracy: 0.9511 - precision: 0.9629 - recall: 0.9384 - auc: 0.9885 - val_loss: 0.0880 - val_tp: 74.0000 - val_fp: 1006.0000 - val_tn: 44482.0000 - val_fn: 7.0000 - val_accuracy: 0.9778 - val_precision: 0.0685 - val_recall: 0.9136 - val_auc: 0.9766
Epoch 68/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1376 - tp: 19177.0000 - fp: 770.0000 - tn: 19769.0000 - fn: 1244.0000 - accuracy: 0.9508 - precision: 0.9614 - recall: 0.9391 - auc: 0.9886 - val_loss: 0.0874 - val_tp: 74.0000 - val_fp: 1006.0000 - val_tn: 44482.0000 - val_fn: 7.0000 - val_accuracy: 0.9778 - val_precision: 0.0685 - val_recall: 0.9136 - val_auc: 0.9768
Epoch 69/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1370 - tp: 19135.0000 - fp: 776.0000 - tn: 19797.0000 - fn: 1252.0000 - accuracy: 0.9505 - precision: 0.9610 - recall: 0.9386 - auc: 0.9888 - val_loss: 0.0861 - val_tp: 74.0000 - val_fp: 986.0000 - val_tn: 44502.0000 - val_fn: 7.0000 - val_accuracy: 0.9782 - val_precision: 0.0698 - val_recall: 0.9136 - val_auc: 0.9769
Epoch 70/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1350 - tp: 19241.0000 - fp: 726.0000 - tn: 19807.0000 - fn: 1186.0000 - accuracy: 0.9533 - precision: 0.9636 - recall: 0.9419 - auc: 0.9893 - val_loss: 0.0848 - val_tp: 74.0000 - val_fp: 971.0000 - val_tn: 44517.0000 - val_fn: 7.0000 - val_accuracy: 0.9785 - val_precision: 0.0708 - val_recall: 0.9136 - val_auc: 0.9771
Epoch 71/1000
20/20 [==============================] - 0s 25ms/step - loss: 0.1325 - tp: 18893.0000 - fp: 760.0000 - tn: 20057.0000 - fn: 1250.0000 - accuracy: 0.9509 - precision: 0.9613 - recall: 0.9379 - auc: 0.9895 - val_loss: 0.0839 - val_tp: 74.0000 - val_fp: 965.0000 - val_tn: 44523.0000 - val_fn: 7.0000 - val_accuracy: 0.9787 - val_precision: 0.0712 - val_recall: 0.9136 - val_auc: 0.9772
Epoch 72/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1323 - tp: 19176.0000 - fp: 711.0000 - tn: 19835.0000 - fn: 1238.0000 - accuracy: 0.9524 - precision: 0.9642 - recall: 0.9394 - auc: 0.9896 - val_loss: 0.0836 - val_tp: 74.0000 - val_fp: 972.0000 - val_tn: 44516.0000 - val_fn: 7.0000 - val_accuracy: 0.9785 - val_precision: 0.0707 - val_recall: 0.9136 - val_auc: 0.9774
Epoch 73/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1322 - tp: 19239.0000 - fp: 716.0000 - tn: 19784.0000 - fn: 1221.0000 - accuracy: 0.9527 - precision: 0.9641 - recall: 0.9403 - auc: 0.9895 - val_loss: 0.0836 - val_tp: 74.0000 - val_fp: 980.0000 - val_tn: 44508.0000 - val_fn: 7.0000 - val_accuracy: 0.9783 - val_precision: 0.0702 - val_recall: 0.9136 - val_auc: 0.9776
Epoch 74/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1318 - tp: 19231.0000 - fp: 744.0000 - tn: 19747.0000 - fn: 1238.0000 - accuracy: 0.9516 - precision: 0.9628 - recall: 0.9395 - auc: 0.9896 - val_loss: 0.0829 - val_tp: 74.0000 - val_fp: 972.0000 - val_tn: 44516.0000 - val_fn: 7.0000 - val_accuracy: 0.9785 - val_precision: 0.0707 - val_recall: 0.9136 - val_auc: 0.9776
Epoch 75/1000
20/20 [==============================] - 0s 25ms/step - loss: 0.1324 - tp: 19321.0000 - fp: 708.0000 - tn: 19727.0000 - fn: 1204.0000 - accuracy: 0.9533 - precision: 0.9647 - recall: 0.9413 - auc: 0.9899 - val_loss: 0.0821 - val_tp: 74.0000 - val_fp: 966.0000 - val_tn: 44522.0000 - val_fn: 7.0000 - val_accuracy: 0.9786 - val_precision: 0.0712 - val_recall: 0.9136 - val_auc: 0.9778
Epoch 76/1000
20/20 [==============================] - 1s 25ms/step - loss: 0.1298 - tp: 19099.0000 - fp: 731.0000 - tn: 19900.0000 - fn: 1230.0000 - accuracy: 0.9521 - precision: 0.9631 - recall: 0.9395 - auc: 0.9900 - val_loss: 0.0820 - val_tp: 74.0000 - val_fp: 974.0000 - val_tn: 44514.0000 - val_fn: 7.0000 - val_accuracy: 0.9785 - val_precision: 0.0706 - val_recall: 0.9136 - val_auc: 0.9779
Epoch 77/1000
20/20 [==============================] - 0s 25ms/step - loss: 0.1284 - tp: 19180.0000 - fp: 700.0000 - tn: 19913.0000 - fn: 1167.0000 - accuracy: 0.9544 - precision: 0.9648 - recall: 0.9426 - auc: 0.9903 - val_loss: 0.0815 - val_tp: 74.0000 - val_fp: 976.0000 - val_tn: 44512.0000 - val_fn: 7.0000 - val_accuracy: 0.9784 - val_precision: 0.0705 - val_recall: 0.9136 - val_auc: 0.9779
Epoch 78/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1308 - tp: 19392.0000 - fp: 763.0000 - tn: 19603.0000 - fn: 1202.0000 - accuracy: 0.9520 - precision: 0.9621 - recall: 0.9416 - auc: 0.9900 - val_loss: 0.0803 - val_tp: 74.0000 - val_fp: 953.0000 - val_tn: 44535.0000 - val_fn: 7.0000 - val_accuracy: 0.9789 - val_precision: 0.0721 - val_recall: 0.9136 - val_auc: 0.9780
Epoch 79/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1301 - tp: 19121.0000 - fp: 729.0000 - tn: 19916.0000 - fn: 1194.0000 - accuracy: 0.9531 - precision: 0.9633 - recall: 0.9412 - auc: 0.9901 - val_loss: 0.0791 - val_tp: 74.0000 - val_fp: 937.0000 - val_tn: 44551.0000 - val_fn: 7.0000 - val_accuracy: 0.9793 - val_precision: 0.0732 - val_recall: 0.9136 - val_auc: 0.9781
Epoch 80/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1279 - tp: 19389.0000 - fp: 717.0000 - tn: 19652.0000 - fn: 1202.0000 - accuracy: 0.9531 - precision: 0.9643 - recall: 0.9416 - auc: 0.9904 - val_loss: 0.0781 - val_tp: 74.0000 - val_fp: 915.0000 - val_tn: 44573.0000 - val_fn: 7.0000 - val_accuracy: 0.9798 - val_precision: 0.0748 - val_recall: 0.9136 - val_auc: 0.9783
Epoch 81/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1291 - tp: 19255.0000 - fp: 709.0000 - tn: 19803.0000 - fn: 1193.0000 - accuracy: 0.9536 - precision: 0.9645 - recall: 0.9417 - auc: 0.9904 - val_loss: 0.0774 - val_tp: 74.0000 - val_fp: 899.0000 - val_tn: 44589.0000 - val_fn: 7.0000 - val_accuracy: 0.9801 - val_precision: 0.0761 - val_recall: 0.9136 - val_auc: 0.9784
Epoch 82/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1271 - tp: 19365.0000 - fp: 672.0000 - tn: 19670.0000 - fn: 1253.0000 - accuracy: 0.9530 - precision: 0.9665 - recall: 0.9392 - auc: 0.9905 - val_loss: 0.0779 - val_tp: 74.0000 - val_fp: 922.0000 - val_tn: 44566.0000 - val_fn: 7.0000 - val_accuracy: 0.9796 - val_precision: 0.0743 - val_recall: 0.9136 - val_auc: 0.9785
Epoch 83/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1256 - tp: 19294.0000 - fp: 698.0000 - tn: 19764.0000 - fn: 1204.0000 - accuracy: 0.9536 - precision: 0.9651 - recall: 0.9413 - auc: 0.9906 - val_loss: 0.0779 - val_tp: 74.0000 - val_fp: 935.0000 - val_tn: 44553.0000 - val_fn: 7.0000 - val_accuracy: 0.9793 - val_precision: 0.0733 - val_recall: 0.9136 - val_auc: 0.9785
Epoch 84/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1239 - tp: 19211.0000 - fp: 682.0000 - tn: 19902.0000 - fn: 1165.0000 - accuracy: 0.9549 - precision: 0.9657 - recall: 0.9428 - auc: 0.9910 - val_loss: 0.0774 - val_tp: 74.0000 - val_fp: 923.0000 - val_tn: 44565.0000 - val_fn: 7.0000 - val_accuracy: 0.9796 - val_precision: 0.0742 - val_recall: 0.9136 - val_auc: 0.9785
Epoch 85/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1261 - tp: 19369.0000 - fp: 708.0000 - tn: 19689.0000 - fn: 1194.0000 - accuracy: 0.9536 - precision: 0.9647 - recall: 0.9419 - auc: 0.9907 - val_loss: 0.0766 - val_tp: 74.0000 - val_fp: 913.0000 - val_tn: 44575.0000 - val_fn: 7.0000 - val_accuracy: 0.9798 - val_precision: 0.0750 - val_recall: 0.9136 - val_auc: 0.9786
Epoch 86/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1250 - tp: 19273.0000 - fp: 676.0000 - tn: 19831.0000 - fn: 1180.0000 - accuracy: 0.9547 - precision: 0.9661 - recall: 0.9423 - auc: 0.9909 - val_loss: 0.0753 - val_tp: 74.0000 - val_fp: 896.0000 - val_tn: 44592.0000 - val_fn: 7.0000 - val_accuracy: 0.9802 - val_precision: 0.0763 - val_recall: 0.9136 - val_auc: 0.9787
Epoch 87/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1244 - tp: 19402.0000 - fp: 660.0000 - tn: 19708.0000 - fn: 1190.0000 - accuracy: 0.9548 - precision: 0.9671 - recall: 0.9422 - auc: 0.9910 - val_loss: 0.0751 - val_tp: 74.0000 - val_fp: 899.0000 - val_tn: 44589.0000 - val_fn: 7.0000 - val_accuracy: 0.9801 - val_precision: 0.0761 - val_recall: 0.9136 - val_auc: 0.9788
Epoch 88/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1246 - tp: 19288.0000 - fp: 700.0000 - tn: 19799.0000 - fn: 1173.0000 - accuracy: 0.9543 - precision: 0.9650 - recall: 0.9427 - auc: 0.9909 - val_loss: 0.0751 - val_tp: 74.0000 - val_fp: 912.0000 - val_tn: 44576.0000 - val_fn: 7.0000 - val_accuracy: 0.9798 - val_precision: 0.0751 - val_recall: 0.9136 - val_auc: 0.9789
Epoch 89/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1244 - tp: 19421.0000 - fp: 666.0000 - tn: 19729.0000 - fn: 1144.0000 - accuracy: 0.9558 - precision: 0.9668 - recall: 0.9444 - auc: 0.9911 - val_loss: 0.0748 - val_tp: 74.0000 - val_fp: 916.0000 - val_tn: 44572.0000 - val_fn: 7.0000 - val_accuracy: 0.9797 - val_precision: 0.0747 - val_recall: 0.9136 - val_auc: 0.9789
Epoch 90/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1234 - tp: 19389.0000 - fp: 675.0000 - tn: 19746.0000 - fn: 1150.0000 - accuracy: 0.9554 - precision: 0.9664 - recall: 0.9440 - auc: 0.9913 - val_loss: 0.0741 - val_tp: 74.0000 - val_fp: 912.0000 - val_tn: 44576.0000 - val_fn: 7.0000 - val_accuracy: 0.9798 - val_precision: 0.0751 - val_recall: 0.9136 - val_auc: 0.9789
Epoch 91/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1227 - tp: 19358.0000 - fp: 670.0000 - tn: 19807.0000 - fn: 1125.0000 - accuracy: 0.9562 - precision: 0.9665 - recall: 0.9451 - auc: 0.9913 - val_loss: 0.0738 - val_tp: 74.0000 - val_fp: 915.0000 - val_tn: 44573.0000 - val_fn: 7.0000 - val_accuracy: 0.9798 - val_precision: 0.0748 - val_recall: 0.9136 - val_auc: 0.9790
Epoch 92/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1210 - tp: 19197.0000 - fp: 658.0000 - tn: 19964.0000 - fn: 1141.0000 - accuracy: 0.9561 - precision: 0.9669 - recall: 0.9439 - auc: 0.9914 - val_loss: 0.0734 - val_tp: 74.0000 - val_fp: 917.0000 - val_tn: 44571.0000 - val_fn: 7.0000 - val_accuracy: 0.9797 - val_precision: 0.0747 - val_recall: 0.9136 - val_auc: 0.9791
Epoch 93/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1218 - tp: 19240.0000 - fp: 715.0000 - tn: 19882.0000 - fn: 1123.0000 - accuracy: 0.9551 - precision: 0.9642 - recall: 0.9449 - auc: 0.9914 - val_loss: 0.0723 - val_tp: 74.0000 - val_fp: 893.0000 - val_tn: 44595.0000 - val_fn: 7.0000 - val_accuracy: 0.9802 - val_precision: 0.0765 - val_recall: 0.9136 - val_auc: 0.9791
Epoch 94/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1197 - tp: 19268.0000 - fp: 719.0000 - tn: 19839.0000 - fn: 1134.0000 - accuracy: 0.9548 - precision: 0.9640 - recall: 0.9444 - auc: 0.9916 - val_loss: 0.0714 - val_tp: 74.0000 - val_fp: 871.0000 - val_tn: 44617.0000 - val_fn: 7.0000 - val_accuracy: 0.9807 - val_precision: 0.0783 - val_recall: 0.9136 - val_auc: 0.9792
Epoch 95/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1206 - tp: 19366.0000 - fp: 655.0000 - tn: 19796.0000 - fn: 1143.0000 - accuracy: 0.9561 - precision: 0.9673 - recall: 0.9443 - auc: 0.9918 - val_loss: 0.0706 - val_tp: 74.0000 - val_fp: 878.0000 - val_tn: 44610.0000 - val_fn: 7.0000 - val_accuracy: 0.9806 - val_precision: 0.0777 - val_recall: 0.9136 - val_auc: 0.9793
Epoch 96/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1185 - tp: 19472.0000 - fp: 621.0000 - tn: 19756.0000 - fn: 1111.0000 - accuracy: 0.9577 - precision: 0.9691 - recall: 0.9460 - auc: 0.9920 - val_loss: 0.0706 - val_tp: 74.0000 - val_fp: 885.0000 - val_tn: 44603.0000 - val_fn: 7.0000 - val_accuracy: 0.9804 - val_precision: 0.0772 - val_recall: 0.9136 - val_auc: 0.9792
Epoch 97/1000
20/20 [==============================] - 0s 25ms/step - loss: 0.1190 - tp: 19415.0000 - fp: 659.0000 - tn: 19753.0000 - fn: 1133.0000 - accuracy: 0.9563 - precision: 0.9672 - recall: 0.9449 - auc: 0.9918 - val_loss: 0.0697 - val_tp: 74.0000 - val_fp: 873.0000 - val_tn: 44615.0000 - val_fn: 7.0000 - val_accuracy: 0.9807 - val_precision: 0.0781 - val_recall: 0.9136 - val_auc: 0.9794
Epoch 98/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1164 - tp: 19510.0000 - fp: 669.0000 - tn: 19656.0000 - fn: 1125.0000 - accuracy: 0.9562 - precision: 0.9668 - recall: 0.9455 - auc: 0.9923 - val_loss: 0.0698 - val_tp: 74.0000 - val_fp: 881.0000 - val_tn: 44607.0000 - val_fn: 7.0000 - val_accuracy: 0.9805 - val_precision: 0.0775 - val_recall: 0.9136 - val_auc: 0.9794
Epoch 99/1000
20/20 [==============================] - 0s 25ms/step - loss: 0.1163 - tp: 19477.0000 - fp: 612.0000 - tn: 19763.0000 - fn: 1108.0000 - accuracy: 0.9580 - precision: 0.9695 - recall: 0.9462 - auc: 0.9923 - val_loss: 0.0690 - val_tp: 74.0000 - val_fp: 863.0000 - val_tn: 44625.0000 - val_fn: 7.0000 - val_accuracy: 0.9809 - val_precision: 0.0790 - val_recall: 0.9136 - val_auc: 0.9796
Epoch 100/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1173 - tp: 19481.0000 - fp: 667.0000 - tn: 19683.0000 - fn: 1129.0000 - accuracy: 0.9562 - precision: 0.9669 - recall: 0.9452 - auc: 0.9920 - val_loss: 0.0679 - val_tp: 74.0000 - val_fp: 848.0000 - val_tn: 44640.0000 - val_fn: 7.0000 - val_accuracy: 0.9812 - val_precision: 0.0803 - val_recall: 0.9136 - val_auc: 0.9797
Epoch 101/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1164 - tp: 19364.0000 - fp: 631.0000 - tn: 19810.0000 - fn: 1155.0000 - accuracy: 0.9564 - precision: 0.9684 - recall: 0.9437 - auc: 0.9923 - val_loss: 0.0674 - val_tp: 74.0000 - val_fp: 829.0000 - val_tn: 44659.0000 - val_fn: 7.0000 - val_accuracy: 0.9817 - val_precision: 0.0819 - val_recall: 0.9136 - val_auc: 0.9798
Epoch 102/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1163 - tp: 19259.0000 - fp: 632.0000 - tn: 19909.0000 - fn: 1160.0000 - accuracy: 0.9563 - precision: 0.9682 - recall: 0.9432 - auc: 0.9923 - val_loss: 0.0671 - val_tp: 74.0000 - val_fp: 834.0000 - val_tn: 44654.0000 - val_fn: 7.0000 - val_accuracy: 0.9815 - val_precision: 0.0815 - val_recall: 0.9136 - val_auc: 0.9762
Epoch 103/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1131 - tp: 19318.0000 - fp: 632.0000 - tn: 19900.0000 - fn: 1110.0000 - accuracy: 0.9575 - precision: 0.9683 - recall: 0.9457 - auc: 0.9928 - val_loss: 0.0676 - val_tp: 74.0000 - val_fp: 860.0000 - val_tn: 44628.0000 - val_fn: 7.0000 - val_accuracy: 0.9810 - val_precision: 0.0792 - val_recall: 0.9136 - val_auc: 0.9763
Epoch 104/1000
20/20 [==============================] - 0s 24ms/step - loss: 0.1166 - tp: 19495.0000 - fp: 648.0000 - tn: 19708.0000 - fn: 1109.0000 - accuracy: 0.9571 - precision: 0.9678 - recall: 0.9462 - auc: 0.9921 - val_loss: 0.0676 - val_tp: 74.0000 - val_fp: 855.0000 - val_tn: 44633.0000 - val_fn: 7.0000 - val_accuracy: 0.9811 - val_precision: 0.0797 - val_recall: 0.9136 - val_auc: 0.9763
Epoch 105/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1180 - tp: 19359.0000 - fp: 690.0000 - tn: 19789.0000 - fn: 1122.0000 - accuracy: 0.9558 - precision: 0.9656 - recall: 0.9452 - auc: 0.9920 - val_loss: 0.0659 - val_tp: 74.0000 - val_fp: 831.0000 - val_tn: 44657.0000 - val_fn: 7.0000 - val_accuracy: 0.9816 - val_precision: 0.0818 - val_recall: 0.9136 - val_auc: 0.9763
Epoch 106/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1173 - tp: 19494.0000 - fp: 647.0000 - tn: 19673.0000 - fn: 1146.0000 - accuracy: 0.9562 - precision: 0.9679 - recall: 0.9445 - auc: 0.9923 - val_loss: 0.0657 - val_tp: 74.0000 - val_fp: 825.0000 - val_tn: 44663.0000 - val_fn: 7.0000 - val_accuracy: 0.9817 - val_precision: 0.0823 - val_recall: 0.9136 - val_auc: 0.9763
Epoch 107/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1164 - tp: 19292.0000 - fp: 631.0000 - tn: 19911.0000 - fn: 1126.0000 - accuracy: 0.9571 - precision: 0.9683 - recall: 0.9449 - auc: 0.9923 - val_loss: 0.0658 - val_tp: 74.0000 - val_fp: 840.0000 - val_tn: 44648.0000 - val_fn: 7.0000 - val_accuracy: 0.9814 - val_precision: 0.0810 - val_recall: 0.9136 - val_auc: 0.9763
Epoch 108/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1167 - tp: 19379.0000 - fp: 675.0000 - tn: 19774.0000 - fn: 1132.0000 - accuracy: 0.9559 - precision: 0.9663 - recall: 0.9448 - auc: 0.9923 - val_loss: 0.0650 - val_tp: 74.0000 - val_fp: 823.0000 - val_tn: 44665.0000 - val_fn: 7.0000 - val_accuracy: 0.9818 - val_precision: 0.0825 - val_recall: 0.9136 - val_auc: 0.9763
Epoch 109/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1136 - tp: 19392.0000 - fp: 589.0000 - tn: 19847.0000 - fn: 1132.0000 - accuracy: 0.9580 - precision: 0.9705 - recall: 0.9448 - auc: 0.9928 - val_loss: 0.0642 - val_tp: 74.0000 - val_fp: 808.0000 - val_tn: 44680.0000 - val_fn: 7.0000 - val_accuracy: 0.9821 - val_precision: 0.0839 - val_recall: 0.9136 - val_auc: 0.9762
Epoch 110/1000
20/20 [==============================] - 0s 23ms/step - loss: 0.1137 - tp: 19231.0000 - fp: 609.0000 - tn: 19981.0000 - fn: 1139.0000 - accuracy: 0.9573 - precision: 0.9693 - recall: 0.9441 - auc: 0.9928 - val_loss: 0.0637 - val_tp: 74.0000 - val_fp: 795.0000 - val_tn: 44693.0000 - val_fn: 7.0000 - val_accuracy: 0.9824 - val_precision: 0.0852 - val_recall: 0.9136 - val_auc: 0.9763
Epoch 111/1000
20/20 [==============================] - ETA: 0s - loss: 0.1133 - tp: 19152.0000 - fp: 619.0000 - tn: 20085.0000 - fn: 1104.0000 - accuracy: 0.9579 - precision: 0.9687 - recall: 0.9455 - auc: 0.9926Restoring model weights from the end of the best epoch.
20/20 [==============================] - 0s 24ms/step - loss: 0.1133 - tp: 19152.0000 - fp: 619.0000 - tn: 20085.0000 - fn: 1104.0000 - accuracy: 0.9579 - precision: 0.9687 - recall: 0.9455 - auc: 0.9926 - val_loss: 0.0634 - val_tp: 74.0000 - val_fp: 789.0000 - val_tn: 44699.0000 - val_fn: 7.0000 - val_accuracy: 0.9825 - val_precision: 0.0857 - val_recall: 0.9136 - val_auc: 0.9762
Epoch 00111: early stopping

"""

#Evaluate metrics
train_predictions_resampled = resampled_model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_resampled = resampled_model.predict(test_features, batch_size=BATCH_SIZE)

resampled_results = resampled_model.evaluate(test_features, test_labels,
                                             batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(resampled_model.metrics_names, resampled_results):
  print(name, ': ', value)
print()

plot_cm(test_labels, test_predictions_resampled)

"""
loss :  0.06738866120576859
tp :  83.0
fp :  1020.0
tn :  55854.0
fn :  5.0
accuracy :  0.9820055365562439
precision :  0.0752493217587471
recall :  0.9431818127632141
auc :  0.9954634308815002

Legitimate Transactions Detected (True Negatives):  55854
Legitimate Transactions Incorrectly Detected (False Positives):  1020
Fraudulent Transactions Missed (False Negatives):  5
Fraudulent Transactions Detected (True Positives):  83
Total Fraudulent Transactions:  88

"""

#Plot the ROC
#plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')

plot_roc("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])
plot_roc("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')

plot_roc("Train Resampled", train_labels, train_predictions_resampled,  color=colors[2])
plot_roc("Test Resampled", test_labels, test_predictions_resampled,  color=colors[2], linestyle='--')
plt.legend(loc='lower right')

#<matplotlib.legend.Legend at 0x7f28100dbcc0>