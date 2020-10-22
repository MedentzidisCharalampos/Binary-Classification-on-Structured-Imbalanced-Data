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
