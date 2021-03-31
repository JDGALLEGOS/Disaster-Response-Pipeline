import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ 
    Import libraries and load datasets.¶

    Import Python libraries
    Load messages.csv into a dataframe and inspect the first few lines.
    Load categories.csv into a dataframe and inspect the first few lines.
    
    Args:
        messages_filepath (string): The file path of the messages csv
        categories_filepath (string): The file path of the categories cv
    Returns:
        df (pandas dataframe): The combined messages and categories df
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on='id')
    #df = pd.concat([messages, categories], axis = 1,join="inner")
    return df
    
def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories[:1]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.applymap(lambda x: x[:-2]).iloc[0,:].tolist()   
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #Convert category values to just 0 or 1. 
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates( inplace=True)
    
    return df
        
def save_data(df, database_filename):
    """Saves the resulting data to a sqlite db
    Args:
        df (pandas dataframe): The cleaned dataframe
        database_filename (string): the file path to save the db
    Returns:
        None
    """  
    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('disaster_response', engine, index=False, if_exists = 'replace')
    engine.dispose()

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print(df.columns.values)
        print('Cleaning data...')
        df = clean_data(df)
        print(df.columns.values)
        
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