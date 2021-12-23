# Project: Disaster Response Pipeline
# Syntax for execution:
# python train_classifier.py <path to sqllite  destination db> <path to the pickle file>
# python train_classifier.py ../data/disaster_response.db classifier.pkl

import sys
import os
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from sqlalchemy import create_engine
import sqlite3
import re
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
import warnings
warnings.filterwarnings('ignore') 

def load_data(database_filepath):
    '''
    INPUT:
    database_filepath - filepath of file within sql database data/DisasterResponseDb.db
    OUTPUT:
    X - array of message data (input data for model)
    Y - array of categorisation of messages (target variables)
    category_names - names of target variables
    Loads data from sql database and extracts information for modelling
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_response', engine)
    X = df['message']
    Y = df.iloc[:, 4:].values
    category_names = df.iloc[:, 4:].columns
    return X, Y, category_names 


def tokenize(text):
    """
    Tokenize and process text data function.
    
    Arguments:
        text -> text messages
    Output:
        clean_tokens -> list of clean tokens
    """

    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url,'urlplaceholder')

    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Build a custom transformer which extracts the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class.
    
    This class extracts the starting verb of a sentence,
    creating a new feature for the ML classifier.
    """

    def starting_verb(self, text):
        sentence_list = sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """
    Build ML pipeline with SVM classifier GridSearch function.
    Output:
        GridSearch output.
    """
    # Pipeline uses KNeighbors Classifier along with a custom made transformer (StartingVerbExtractor)
    pipelineKN = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb_transformer', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(estimator=KNeighborsClassifier()))
    ])
    
    param_gridKN = {'clf__estimator__n_neighbors': (1,10),
                    'clf__estimator__leaf_size': (20,1)}

    cv = GridSearchCV(estimator=pipelineKN, param_grid=param_gridKN)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model function.
    Arguments:
        model -> sklearn model
        X_test -> Features for the test set
        Y_test -> Labels for the test set
        category_names -> list of category names
    """

    Y_pred = model.predict(X_test)

    # Classification report
    print("Classification report")
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Save model function.
    
    This function saves trained model as Pickle file.
    
    Arguments:
        model -> sklearn pipeline object
        model_filepath -> destination path to save .pkl file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()