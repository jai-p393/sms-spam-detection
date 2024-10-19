# SMS Spam Detection - NLP Project

This repository contains a project for detecting SMS spam using Natural Language Processing (NLP) techniques and machine learning algorithms. The dataset used is **SMS Spam Collection v.1**, which includes labeled SMS messages as either "ham" (legitimate) or "spam."

## Project Structure

- **dataset/**: This folder contains the `SMSSpamcollection.csv` dataset used in the project.
- **BOW_implementation.ipynb**: Implements Bag of Words (BOW) for word vectorization and uses Multinomial Naive Bayes for classification.
- **TFIDF_implementation.ipynb**: Implements TF-IDF for word vectorization and uses Multinomial Naive Bayes for classification.
- **Word2Vec_implementation_I.ipynb**: Implements Word2Vec (without removing stopwords) for word vectorization and uses RandomForestClassifier for classification.
- **Word2Vec_implementation_II.ipynb**: Implements Word2Vec (with stopwords removed) for word vectorization and uses RandomForestClassifier for classification.
- **requirements.txt**: Lists all dependencies required to run the project.

## Notebooks Overview

1. **BOW_implementation.ipynb**:
   - Preprocessing: Lowercasing, removing special characters, removing stopwords, and lemmatization.
   - Word Vectorization: Bag of Words (BOW).
   - Classifier: Multinomial Naive Bayes.
   - **Accuracy**: 98.65%.

2. **TFIDF_implementation.ipynb**:
   - Preprocessing: Lowercasing, removing special characters, removing stopwords, and lemmatization.
   - Word Vectorization: TF-IDF.
   - Classifier: Multinomial Naive Bayes.
   - **Accuracy**: 97.21%.

3. **Word2Vec_implementation_I.ipynb**:
   - Preprocessing: Using `gensim.utils.simple_preprocess()` (without stopword removal).
   - Word Vectorization: Word2Vec (trained on the training set).
   - Classifier: RandomForestClassifier.
   - **Accuracy**: 96.85%.

4. **Word2Vec_implementation_II.ipynb**:
   - Preprocessing: Lowercasing, removing special characters, removing stopwords, and lemmatization.
   - Word Vectorization: Word2Vec (trained on the training set).
   - Classifier: RandomForestClassifier.
   - **Accuracy**: 96.31%.

## Dataset

The dataset used in this project, **SMS Spam Collection v.1**, contains a collection of SMS messages categorized as either spam or ham. It is stored in the `dataset/` directory.

- **Total Records**: 5,574 messages.
  - **Ham**: 4,827 messages (86.6%)
  - **Spam**: 747 messages (13.4%)


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

## Future Improvements

- Explore deep learning models like LSTM or transformers.
- Experiment with more advanced embeddings (GloVe, FastText).
- Perform hyperparameter tuning for improved accuracy.







