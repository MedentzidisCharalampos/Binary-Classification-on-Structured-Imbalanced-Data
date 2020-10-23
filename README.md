# Binary-Classification-on-Structured-Imbalanced-Data
Classification on the  Credit Card Fraud Detection  that is highly imablanced.

# The dataset
The Credit Card Fraud Detection (https://www.kaggle.com/mlg-ulb/creditcardfraud) dataset aims to detect a mere 492 fraudulent transactions from 284,807 transactions in total. 

# Data Preprocessing
The raw data has a few issues. First the Time and Amount columns are too variable to use directly. Drop the Time column take the log of the Amount column to reduce its range.
Normalize the input features. We set the mean to 0 and standard deviation to 1.

# Data distribution

![alt text](https://github.com/MedentzidisCharalampos/Binary-Classification-on-Structured-Imbalanced-Data/blob/main/positive_distribution.png =1200x1200)
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
1. Build the model: The model is fit using a larger than default batch size of 2048, this is important to ensure that each batch has a decent chance of containing a few positive samples. If the batch size was too small, they would likely have no fraudulent transactions to learn from.
2. Set the correct initial bias
3. Checkpoint the initial weights.
4. Confirm that the bias fix helps.  
![alt text](https://github.com/MedentzidisCharalampos/Binary-Classification-on-Structured-Imbalanced-Data/blob/main/loss_convergence.png)
5. Train the model for 100 epochs and early stopping to avoid overfitting.
6. Check training history: Produce plots of your model's accuracy and loss on the training and validation set.
![alt text](https://github.com/MedentzidisCharalampos/Binary-Classification-on-Structured-Imbalanced-Data/blob/main/training_epochs.png)  
6. Evaluate metrics: We use a confusion matrix to summarize the actual vs. predicted labels where the X axis is the predicted label and the Y axis is the actual label.  
![alt text](https://github.com/MedentzidisCharalampos/Binary-Classification-on-Structured-Imbalanced-Data/blob/main/confusio_matrix_.png)
If the model had predicted everything perfectly, this would be a diagonal matrix where values off the main diagonal, indicating incorrect predictions, would be zero. In this case the matrix shows that we have relatively few false positives, meaning that there were relatively few legitimate transactions that were incorrectly flagged. However, we would likely want to have even fewer false negatives despite the cost of increasing the number of false positives. This trade off may be preferable because false negatives would allow fraudulent transactions to go through, whereas false positives may cause an email to be sent to a customer to ask them to verify their card activity.  
7. Plot the ROC.    
![alt text](https://github.com/MedentzidisCharalampos/Binary-Classification-on-Structured-Imbalanced-Data/blob/main/tp_fp.png)  
It looks like the precision is relatively high, but the recall and the area under the ROC curve (AUC) aren't as high as you might like. It is important to consider the costs of different types of errors in the context of the problem we care about. In our case, a false negative (a fraudulent transaction is missed) may have a financial cost, while a false positive (a transaction is incorrectly flagged as fraudulent) may decrease user happiness.

# Class Weights
1. Calculate class weights. The goal is to identify fraudulent transactions, but we don't have very many of those positive samples to work with, so we would want to have the classifier heavily weight the few examples that are available.
2. Train a model with class weights.  
3. Check training history.  
![alt text](https://github.com/MedentzidisCharalampos/Binary-Classification-on-Structured-Imbalanced-Data/blob/main/weighted_training.png)
4. Evaluate Metrics.  
![alt text](https://github.com/MedentzidisCharalampos/Binary-Classification-on-Structured-Imbalanced-Data/blob/main/confusion_weighted.png)  
 With class weights the accuracy and precision are lower because there are more false positives, but conversely the recall and AUC are higher because the model also found more true positives. Despite having lower accuracy, this model has higher recall (and identifies more fraudulent transactions).  
 5. Plot the ROC.
 ![alt text](https://github.com/MedentzidisCharalampos/Binary-Classification-on-Structured-Imbalanced-Data/blob/main/roc_weighted.png)

# Oversampling
1. Oversample the minority class.
2. Train on the oversampled data.
3. Check training history with early stopping to avoid overfit.
![alt text](https://github.com/MedentzidisCharalampos/Binary-Classification-on-Structured-Imbalanced-Data/blob/main/plot_metric_oversampling_early_Stop.png)
4. Evaluate metrics(https://github.com/MedentzidisCharalampos/Binary-Classification-on-Structured-Imbalanced-Data/blob/main/plot_metric_oversampling_early_Stop.png)
5. Plot the ROC.  
![alt text](https://github.com/MedentzidisCharalampos/Binary-Classification-on-Structured-Imbalanced-Data/blob/main/roc_oversampling.png)
