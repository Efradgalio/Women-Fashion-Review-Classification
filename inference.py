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
import warnings
import joblib
import torch
import nltk
import re

# Classic Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Transformer Model
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline


# Load Logistic Regression Model
logreg = joblib.load('./models/logreg_v1.joblib')

# Load the pre-trained BERT model with the text classification pipeline
classifier = pipeline("text-classification", model="Frags/finetuned-bert-women-fashion", tokenizer="bert-base-uncased")

def predict(text, text2):

    # Make predictions for Logisict Regression
    logreg_pred = logreg.predict(text)
    logreg_proba = logreg.predict_proba(text)[:, 1]

    # Make predictions for BERT
    predictions = classifier(list(text2.values))

    return logreg_pred, logreg_proba, predictions