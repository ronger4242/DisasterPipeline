import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import pickle

def load_data(database_filepath):
    '''
    load data from database_filepath
    Args: database_file: path to the database file
    Returns: 
        X: feature
        Y: target
        category_names: same as the columns for Y
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table("DisasterData", engine)
    df = df.iloc[:1000]#since the dataset is too big to run, only the first 1000 rows were used.
    X = df['message']
    Y = df.iloc[:,5:]
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text):
    '''
    Process text data: extraction, tokenization, lemmatization, stopwords removal
    Args: 
        text: input text
    Returns:
        clean tokens
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
    Find the best machine learning pipeline model with hyperparameter searching
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    parameters = {
        'vect__ngram_range':((1,1), (1,2)),
        'clf__estimator__n_estimators': [5,10]}    
    cv = GridSearchCV(pipeline, parameters)
    return cv
    

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model training results
    Args:
        model: Training model
        X_test: Test features data (pd.DataFrame)
        Y_test: Test targets data (pd.DataFrame)
        category_names: Target names
    '''
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns = Y_test.columns)
    for column in category_names:
        print('\n ---- {} ----\n{}\n'.format(column,classification_report(Y_test[column],Y_pred[column])))
    

def save_model(model, model_filepath):
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