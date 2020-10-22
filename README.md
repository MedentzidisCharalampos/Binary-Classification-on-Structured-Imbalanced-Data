# Binary-Classification-on-Structured-Imbalanced-Data
Classification on the  Credit Card Fraud Detection  that is highly imablanced.

# The dataset
The Credit Card Fraud Detection (https://www.kaggle.com/mlg-ulb/creditcardfraud) dataset aims to detect a mere 492 fraudulent transactions from 284,807 transactions in total. 

# Data Preprocessing
The raw data has a few issues. First the Time and Amount columns are too variable to use directly. Drop the Time column take the log of the Amount column to reduce its range.
Normalize the input features. We set the mean to 0 and standard deviation to 1.

# Data distribution

![alt text](https://github.com/MedentzidisCharalampos/Binary-Classification-on-Structured-Imbalanced-Data/blob/main/positive_distribution.png)
![alt text](https://github.com/MedentzidisCharalampos/Binary-Classification-on-Structured-Imbalanced-Data/blob/main/negative_distribution.png)

1. The most of the values are concentrated in range [-2, 2] which make sense as we have normalize the data.
2. The positive examples contain a much higher rate of extreme values.

# Architecture of the Model
1. A densly connected hidden layer with 16-units and Relu activation function.
2. A dropout layer with propability of 0.5 to reduce overfitting.
3. An output sigmoid layer that returns the probability of a transaction being fraudulent.

# Metrics
1. `False negatives` and `false positives` are samples that were incorrectly classified.
2. `True negatives` and `true positives` are samples that were correctly classified.
3. `Accuracy` is the percentage of examples correctly classified.
4. `Precision` is the percentage of predicted positives that were correctly classified.
5. `Recall` is the percentage of actual positives that were correctly classified.
6. `AUC` refers to the Area Under the Curve of a Receiver Operating Characteristic curve (ROC-AUC). This metric is equal to the probability that a classifier will rank a random positive sample higher than a random negative sample.

Accuracy is not a helpful metric for this task. We can 99.8%+ accuracy on this task by predicting False all the time.

# Baseline Model
1. The model is fit using a larger than default batch size of 2048. If the batch size was too small, they would likely have no fraudulent transactions to learn from.
2. The initial bias is setted properly to help initial convergence.
![alt text](https://github.com/MedentzidisCharalampos/Binary-Classification-on-Structured-Imbalanced-Data/blob/main/loss_convergence.png)
3. Checkpoint the initial weights.
4. Train the model for 20 epochs and early stopping to avoid overfitting.
5. Produce plots of your model's accuracy and loss on the training and validation set.
![alt text](https://github.com/MedentzidisCharalampos/Binary-Classification-on-Structured-Imbalanced-Data/blob/main/training_epochs.png)
