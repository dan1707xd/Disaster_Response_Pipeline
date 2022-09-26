import sys
# import libraries
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('omw-1.4')
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt


def database_gen():
    # load messages dataset
    messages = pd.read_csv('disaster_messages.csv')
    # messages.head()
    # messages.isna().sum()
    # load categories dataset
    categories = pd.read_csv('disaster_categories.csv')
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
    engine = create_engine('sqlite:///Disaster_Data.db')
    print("---Creating Database---")
    df1.to_sql('Messages-Categories', engine, index=False, if_exists='replace')
    print("---Database created---")


def main():
    database_gen()


if __name__ == '__main__':
    main()