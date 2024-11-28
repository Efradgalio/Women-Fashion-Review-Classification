from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from IPython.display import display
from wordcloud import WordCloud
from collections import Counter
from joblib import dump
from joblib import load
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import warnings
import torch
import nltk
import re
import os

# NLTK
from nltk import FreqDist
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')
nltk.download('wordnet')
nltk.download('punkt_tab')


stop_words = set(stopwords.words('english'))
# Remove 'but' and 'this' as we know from before these 2 words are important.
stop_words.remove('but')
stop_words.remove('this')

# Load the TF-IDF model back
loaded_tfidf = joblib.load('./models/tfidf_fit_transform.joblib')

def remove_parentheses(text):
    pattern = r'\(.*?\)'
    text = re.sub(pattern, "", text)
    return text

def remove_numbers(text):
    text = re.sub(r'\d+', '', text)
    return text

def remove_punctuation(text):
    punctuation_pattern = r'[^\w\s]'
    cleaned_text = re.sub(punctuation_pattern, ' ', text)
    return cleaned_text

def remove_extra_spaces(text):
    extra_spaces_pattern = r"\s+"
    cleaned_text = re.sub(extra_spaces_pattern, " ", text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def remove_special_characters(text):
    cleaned_text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    return cleaned_text

def lowercase(text):
    lowercase_text = text.lower()
    return lowercase_text

def remove_stopwords(text, stopwords=stop_words):
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    cleaned_text = ' '.join(filtered_tokens)
    return cleaned_text


def clean_preprocessing(df, col, prep_col):
    # Apply preprocessing functions to article column
    df[prep_col] = df[col].apply(remove_parentheses)
    df[prep_col] = df[prep_col].apply(remove_numbers)
    df[prep_col] = df[prep_col].apply(remove_punctuation)
    df[prep_col] = df[prep_col].apply(remove_extra_spaces)
    df[prep_col] = df[prep_col].apply(remove_special_characters)
    df[prep_col] = df[prep_col].apply(lowercase)
    df[prep_col] = df[prep_col].apply(remove_stopwords)
    return df

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatization(text):
    # Step 1: Tokenize the text
    tokens = word_tokenize(text)

    # Step 2: Lemmatize each word
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(lemmatized_tokens)

# Put TFIDF Model fit inside this code, because somehow the tfidf_fit_transform.joblib gaves an error
# That says "idf vector is not fitted". I don't have much time, so will use brute force for now.
data_X_train = pd.read_csv('./dataset/data_X_train.csv')
x_train = data_X_train['Review Text Preprocessed Lemma']
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(x_train)


def full_preprocessing(text):
    # Create a DataFrame with the review as a single row, because preprocessing function needs a dataframe input.
    df = pd.DataFrame([text], columns=['Review'])
    df = clean_preprocessing(df, 'Review', 'Review Preprocessed')
    df['Review Preprocessed Lemma'] = df['Review Preprocessed'].apply(lemmatization)

    # Loaded the TF-IDF Model
    # new_data_tfidf = loaded_tfidf.transform(df['Review Preprocessed Lemma'])
    new_data_tfidf = tfidf_vectorizer.transform(df['Review Preprocessed Lemma'])

    return new_data_tfidf, df['Review Preprocessed Lemma']