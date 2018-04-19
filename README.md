# CreditCardFraudDetection
An attempt to detect fraud in online transaction in deep neural network using pytorch

This is a feed forward neural network built using pytorch to detect fraudulent credit card transaction. The data set is not included in this 
repo. (The link to the dataset is given below)

credit_card_fraud_detector.py loads the dataset, and preprocesses it, and feeds it to the neural network. The one problem here is that the 
dataset is highly skewed. So while meausring the accuracy of the model you need a different method such as F1 score which is not included in
this repo. Feel free to add it.

Enjoy!!

Here's the data set download link : https://www.kaggle.com/mlg-ulb/creditcardfraud/downloads/creditcardfraud.zip/3
