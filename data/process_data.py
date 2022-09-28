import sys
import pandas as pd
from sqlalchemy import create_engine



def database_gen(messages_filepath,categories_filepath):
    '''
    - Loads the csv files for the messages and categories and does the necessary cleaning steps required
    before the NLP processing
    - Saves the dataframe to an SQLlite Database
    Args: messages_filepath, categories_filepath

    --> Creates Database with 'Messages-Categories' as table name

    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # messages.head()
    # messages.isna().sum()
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # categories.head()
    # categories.isna().sum()
    # merge datasets
    df = pd.merge(messages, categories, on='id')
    # df.head()
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)
    # categories.head()
    # select the first row of the categories dataframe
    row = categories.loc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    # print(category_colnames)
    # rename the columns of `categories`
    categories.columns = category_colnames
    # categories.head()
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: int(x[-1]))

    # convert column from string to numeric
    # categories[column] =
    # categories.head()
    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])

    # df.head()
    # concatenate the original dataframe with the new `categories` dataframe
    df1 = pd.concat([df, categories], axis=1, sort='False')
    # df1
    # check number of duplicates
    duplicates = df1.duplicated().value_counts()
    # duplicates
    # drop duplicates
    df1 = df1.drop_duplicates()
    # check number of duplicates
    duplicates_check = df1.duplicated().value_counts()
    # duplicates_check
    #Convert the related column to binary
    df1['related'] = df1['related'].astype('str').str.replace('2', '1')
    df1['related'] = df1['related'].astype('int')
    engine = create_engine('sqlite:///Disaster_Data.db')
    print("---Creating Database---")
    df1.to_sql('Messages-Categories', engine, index=False, if_exists='replace')
    print("---Database created---")



if __name__ == '__main__':
    try:
        messages_filepath, categories_filepath = "disaster_messages.csv","disaster_categories.csv"
        database_gen(messages_filepath, categories_filepath)
    except FileNotFoundError:
        print('Review and make sure the datasets filepaths are correctly defined')
