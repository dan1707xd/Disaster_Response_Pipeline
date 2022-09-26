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

def load_data():
    '''Loads data from database'''
    engine = create_engine('sqlite:///../data/Disaster_Data.db')
    df = pd.read_sql_table('Messages-Categories',engine)
    X = df.message
    y = df.iloc[:,4:]
    #print(len(X))
    return X, y

def tokenize(text):
    '''Simple tokenize function for countvectorizer'''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens=list(lemmatizer.lemmatize(tok).lower().strip() for tok in tokens)
    return clean_tokens


from sklearn.base import BaseEstimator, TransformerMixin

class ExtraFeatures(BaseEstimator, TransformerMixin):
    '''Custom transformer that computes 4 additional features:
       1. character count
       2. word count
       3. sentence count
       4. capital word count'''

    def extra_features(self, text):
        sentence_list = nltk.sent_tokenize(text)
        word_list = nltk.word_tokenize(text)
        word_count = len(word_list)
        sent_count = len(sentence_list)
        char_count = len(text)
        capital_word_count = sum(map(str.isupper, text.split()))
        return char_count, word_count, sent_count, capital_word_count

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.extra_features)
        X_df = pd.DataFrame(X_tagged)

        return pd.DataFrame(X_df['message'].tolist(), index=X_df.index,
                            columns=['char_count', 'word_count', 'sent_count', 'capital_word_count'])


from sklearn.base import BaseEstimator, TransformerMixin
import re


class Word2vec(BaseEstimator, TransformerMixin):
    '''Custom transformer that uses the skip gram algorithm from
    word2vec to generate features based on train corpus and a message by
    message basis'''

    def word_list(self, text):
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)
        text = text.lower()
        sentence_list = nltk.sent_tokenize(text)
        word_list = [nltk.word_tokenize(x) for x in sentence_list]
        return word_list

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.word_list)
        X_df = pd.DataFrame(X_tagged)
        vocab = []
        for x in X_df.message:
            vocab.extend(x)
        # print(vocab)
        from gensim.models import word2vec, Word2Vec

        num_features = 300
        min_word_count = 3
        num_workers = 4
        context = 8
        downsampling = 1e-3

        # Initialize and train the model
        W2Vmodel = Word2Vec(sentences=vocab, sg=1, hs=0, workers=num_workers, vector_size=num_features,
                            min_count=min_word_count, window=context, sample=downsampling, negative=5, epochs=6)
        model = W2Vmodel
        model_voc = set(model.wv.key_to_index.keys())
        sent_vector = np.zeros(model.vector_size, dtype="float32")

        dummy = np.zeros((len(X_df), num_features))
        col_names = list('w2v' + str(x) for x in range(num_features))
        df = pd.DataFrame(dummy, columns=col_names)

        for k, sentence in enumerate(X_df.message):
            sent_vector = np.zeros(model.vector_size, dtype="float32")
            for sentence in sentence:
                for x in sentence:
                    words = []
                    words.append(x)

                    nwords = 0
                    for word in words:
                        if word in model_voc:
                            sent_vector += model.wv[word]
                            nwords += 1.
                        if nwords > 0:
                            sent_vector /= nwords
            # print(k, sent_vector)
            df.iloc[k] = sent_vector
        return df


def model_1():
    '''Random Forest with count_vect, tfidf only'''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {
        'vect__ngram_range': [(1, 2)],
        'clf__estimator__n_estimators': [50],
        'clf__estimator__min_samples_split': [3]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=3,cv=3)

    return cv


def model_2():
    '''Random Forest with all features'''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('1_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('2_pipeline', Pipeline([
                ('extra', ExtraFeatures()),
                ('min_max', MinMaxScaler())
            ])),

            ('w2v', Word2vec())

        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'features__1_pipeline__vect__ngram_range': ((1, 2), (1, 3)),
        'clf__estimator__n_estimators': [50]
        # ,'clf__estimator__min_samples_split': [2, 3, 4]
    }

    parameters2 = {
        'features__1_pipeline__vect__ngram_range': ((1, 1), (1, 2), (1, 3)),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters2, n_jobs=-1, verbose=1)
    # cv = pipeline

    return cv


def model_3():
    '''MNB with all features'''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('1_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('2_pipeline', Pipeline([
                ('extra', ExtraFeatures()),
                ('min_max', MinMaxScaler())
            ])),

            ('w2v', Word2vec())

        ])),

        ('clf', MultiOutputClassifier(MultinomialNB()))
    ])

    parameters = {
        'features__1_pipeline__vect__ngram_range': [(1, 1), (1, 2), (1, 3)],  # reduce after confirm
        'clf__estimator__alpha': [0.05, 0.1, 1, 5, 10],

    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=1)
    # cv = pipeline

    return cv


def display_results(cv, y_test, y_pred, name):
    import matplotlib.pyplot as plt
    precision = []
    recall = []
    fscore = []
    support = []
    # print("\nBest Parameters:", cv.best_params_)
    for i in range(len(y_test.columns)):
        # print(str(i)+": "+ y_test.columns[i]+"\n"+classification_report(y_test.iloc[:,i], y_pred[:,i]))
        a, b, c, d = score(y_test.iloc[:, i], y_pred[:, i], average='weighted')
        precision.append(a)
        recall.append(b)
        fscore.append(c)
        support.append(d)
    accuracy = (y_pred == y_test).mean()
    # print(len(fscore))
    # print(len(accuracy))
    # print(fscore)
    a1 = np.array(accuracy)
    a2 = np.array(fscore)
    index = y_test.columns
    df = pd.DataFrame({'Accuracy' + " " + name: a1,
                       'F1_Score' + " " + name: a2}, index=index)

    # ax = df.plot.bar()

    return df


def main():
    #database_gen()
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    '''3 models have been defined (see above):
        1. model_1:random forest classifier without the additional word2vec transformer and smaller custom transformer
        2. model_2:random forest classifier with the additional word2vec transformer and smaller custom transformer in 
            pipeline
        3. model_3: multinomial naive bayes with all the features
    Sidenote: the word2vec transformer's additional features makes the model much better, I play tested a lot and 
                compared the scores.
                You can check and verify using the Jupyter notebook that contains all the code
                Warning: training times for model_2, 3 were upwards of 12 hours(at least for my setup)'''

    model = model_1()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    import pickle
    filename = 'model_1_final.sav'
    pickle.dump(model, open(filename, 'wb'))


if __name__ == '__main__':
    main()
