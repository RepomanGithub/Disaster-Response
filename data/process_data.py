# Project: Disaster Response Pipeline
# Syntax for execution:
# python process_data.py <path to messages csv file> <path to categories csv file> <path to sqllite  destination db>
# python process_data.py disaster_messages.csv disaster_categories.csv disaster_response.db


# import libraries
import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets and merge using common id function.
    Arguments:
        messages_filepath -> csv path of file containing messages
        categories_filepath -> csv path of file containing categories
    Output:
        df -> combined dataset of messages and categories
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    """
    Clean categories data function.
    Arguments:
        df -> combined dataset of messages and categories
    Output:
        df -> combined dataset of messages and categories cleaned
    """
    
    # new data frame with split the values in the categories column on the ';' 
    categories = df.categories.str.split(';', expand=True)
    
    # use the first row of categories dataframe to create column names for the categories data
    row = categories.iloc[0]
    category_colnames = row.map(lambda x: str(x)[:-2])
    categories.columns = category_colnames
    
    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df` and replaces it
    df.drop(columns=['categories'], inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # removes all N/A from the data
    df = df.dropna()
    
    # removes all duplicates from the data
    df.drop_duplicates(inplace=True)
    
    # Remove child_alone as it has all zeros
    df = df.drop(['child_alone'],axis=1)
    
    # There is a category 2 in 'related' column. This could be an error. 
    # In the absense of any information, we assume it to be 1 as the majority class.
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    
    return df

def save_data(df, database_filename):
    """Save into  SQLite database.
    
    inputs:
    df: dataframe. Dataframe containing cleaned version of merged message and 
    categories data.
    database_filename: string. Filename for output database.
       
    outputs:
    None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_response', engine, if_exists='replace', index=False)
    
def main():
    """
    Main function that execute the data processing functions. There are three primary actions taken by this function:
        1) Load messages and categories datasets and merge them
        2) Clean categories data
        3) Save the clean dataset into an sqlite database function
    """
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