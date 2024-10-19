# SMS Spam Detection - NLP Project

## Overview

This project focuses on detecting spam messages in SMS data using Natural Language Processing (NLP) techniques. The dataset used is the **SMS Spam Collection v.1**, which contains 5,574 SMS messages tagged as "ham" (legitimate) or "spam". The goal of this project is to preprocess the text data and apply machine learning algorithms to classify the messages.

## Dataset

The dataset used is the **SMS Spam Collection v.1**, which can be found [here](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/). It contains:

- 4,827 "ham" (legitimate) messages.
- 747 "spam" messages.

## Approach

### 1. Data Preprocessing
The raw text data is cleaned and prepared using the following steps:
- **Lowercasing**: Convert all text to lowercase.
- **Removing special characters**: Eliminate non-alphabetic characters.
- **Stopwords removal**: Remove common English words that don't add significant meaning (using NLTK's stopwords list).
- **Lemmatization**: Normalize words to their root forms using `WordNetLemmatizer`.

### 2. Feature Extraction
The cleaned text is transformed into numerical features using the **Bag of Words (BOW)** model, which converts text into a matrix of token counts.

### 3. Model Training
A **Multinomial Naive Bayes** algorithm was used for classification. The model achieved an impressive accuracy of **99%** on the test data.

### 4. Future Enhancements
Further improvements to the model will include:
- Using **TF-IDF** (Term Frequency-Inverse Document Frequency) for feature extraction.
- Applying **Word2Vec (Average Word2Vec)** for better representation of text data.
- Experimenting with other machine learning algorithms for better performance.


## Installation and Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/jai-p393/sms-spam-detection.git
   cd sms-spam-detection
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the Jupyter notebook to see the analysis and model implementation:
   ```bash
   jupyter notebook spam-ham(using BOW).ipynb

## Results

- Model: **Multinomial Naive Bayes**
- Accuracy: **99%**

### Classification Report
![image](https://github.com/user-attachments/assets/be81967b-76d0-4515-9f06-81116c29812d)






