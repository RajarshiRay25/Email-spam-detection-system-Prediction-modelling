# Email-spam-detection-system-Prediction-modelling
Using NLP and ML model to classify between ham and spam . This architecture uses TF-IDF to perform vectorisation. This code can be referenced into similar projects requiring classification modelling on text data.
Certainly! Here's a more detailed Markdown template for your GitHub README:


# Email Spam Detection System

## Overview

This project implements an email spam detection system using Natural Language Processing (NLP) techniques. The system is designed to classify emails into spam and non-spam categories based on the content of the messages. The implementation involves data preprocessing, text processing with NLP, exploratory data analysis (EDA), and training multiple machine learning models for classification.

## Code Structure

### 1. Data Preprocessing

- Loading the dataset from a CSV file (`spam.csv`).
- Handling null values and removing duplicate entries.
- Renaming columns for clarity (`'v1'` to `'mail_type'`, `'v2'` to `'message'`).
- Mapping categorical labels to numerical values (`'ham'` to `0`, `'spam'` to `1`).

### 2. Text Processing with NLP

- Utilizing NLTK library for NLP operations.
- Removing non-alphabetic characters, converting to lowercase, and stemming for feature extraction.
- Vectorizing text data using the Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer.

### 3. Exploratory Data Analysis (EDA)

- Visualizing the distribution of email categories using seaborn's countplot.
- Addressing the issue of class imbalance in the dataset.

### 4. Model Training

- Splitting the dataset into training and testing sets.
- Training multiple machine learning models:
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM) with a linear kernel.

### 5. Issue Faced

- During testing, encountered a "ValueError: X has 18 features, but SVC is expecting 6221 features as input."

### 6. Root Cause Analysis

- Mismatch in features between training and testing datasets due to a new instance of TfidfVectorizer during testing.

### 7. Resolution

- Modified code to store the training vectorizer (`vectorizer_for_training`) and used it during testing.
- Ensured consistency in vocabulary and feature dimensions between training and testing data.

## Code Snippet


#### Use the training vectorizer on the training dataset
X_training_set, vectorizer_for_training = vectorizer(text_data_train_file)

#### Use the training vectorizer on the testing dataset
X_testing_set, _ = vectorizer(testing_data_file, training_vectorizer=vectorizer_for_training)

## Dependencies

- Python 3.x
- scikit-learn
- NLTK
- pandas
- numpy
- matplotlib
- seaborn

## How to Run

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the main script.

## Results

- Provide insights into the accuracy scores and performance metrics of each trained model.
- Showcase any visualizations generated during the EDA process.

## Future Improvements

- Discuss potential enhancements, optimizations, or additional features that could be added to improve the system's performance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Feel free to customize this template further based on your project's specifics and any additional details you want to include.
