# Spam Mail Detection

### Overview:
This project implements a Spam Mail Detection system using Machine Learning. It classifies emails as either spam or ham (non-spam) using a Support Vector Machine (SVM) model trained on a dataset of email messages.

### Features:
- Preprocesses email data by handling missing values and renaming columns.
- Converts text data into numerical features using TF-IDF vectorization.
- Uses a Support Vector Machine (SVM) classifier for spam detection.
- Evaluates model performance using accuracy scores.
- Allows users to input an email and predict whether it is spam or ham.

### How It Works:
- The dataset is loaded and preprocessed to handle missing values.
- The text data is converted into numerical features using TF-IDF vectorization.
- The data is split into training and testing sets.
- A Support Vector Machine (SVM) model is trained on the training set.
- The model is evaluated on both training and test data.
- A user can input an email, and the model predicts whether it is spam or ham.

### Dataset:
The model is trained on the spam.csv dataset, which contains labeled email messages categorized as spam or ham.
