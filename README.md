# Spam Mail Detection

### Overview:
This project implements a Spam Mail Detection system using Machine Learning. It classifies emails as either spam or ham (non-spam) using a Support Vector Machine (SVM) model trained on a dataset of email messages.

### Features:
✅ Preprocesses email data by handling missing values and renaming columns.  
✅ Converts text into numerical features using TF-IDF vectorization.  
✅ Uses a Support Vector Machine (SVM) classifier for spam detection.  
✅ Evaluates model performance using accuracy scores.  
✅ Interactive UI with Streamlit to check if an email is spam.

### How It Works:
1) The dataset is loaded and preprocessed to handle missing values.
2) The text data is converted into numerical features using TF-IDF vectorization.  
3) The data is split into training and testing sets.
4) An SVM model is trained on the training set.
5) The model is evaluated on both training and test data.
6) Users can interact with the model using a Streamlit-based UI:
- Enter the email content in the text box.
- Click the "Check Email" button.
- Get an instant Spam/Ham prediction.

### Dataset:
The model is trained on the spam.csv dataset, which contains labeled email messages categorized as spam or ham.

### How to Run the Streamlit UI :
To run the interactive Streamlit UI, execute the following command in your terminal:
```bash
streamlit run webapp.py
```

