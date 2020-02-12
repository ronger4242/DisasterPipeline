import sys
import pandas as pd
import numpy as np
import sqlite3
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load two CSV files
    Args:
        message_filepath = path to the message file
        categories_filepath = path to the categories file
    Return df: a merged pandas dataframe contains messages and categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id')
    return df


def clean_data(df):
    '''
    Cleans the dataframe for machine learning pipeline
    Args: 
        df: a dirty dataframe
    Returns:
        df: a tidy dataframe
    '''
    categories = df['categories'].str.split(';', expand = True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str.split('-').str.get(1)
        categories[column] = pd.to_numeric(categories[column])
        categories[column] = categories[column].astype('int')
    df = df.drop('categories', axis = 1)
    df = pd.concat([df,categories], axis = 1)
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    '''
    save the clean dataset into a sqlite database.
    Args:
        df: the pandas dataframe to be saved
        database_filename: the name for the sqlite database file
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterData', con = engine, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()