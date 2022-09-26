# Disaster Response Machine Learning Pipeline
# Project Motivation/Details
The work accomplished for this project attempts to explore a sophisticated and novel approach to a NLP Machine Learning problem. The model's job is to classsify messages typically received during disasters. The classification categories, for example, include aid-related, medical help, lack of water etc. A high precision (or recall for certain categories) model is hugely beneficial to a disaster relief agency, which in times of crisis, will be able to allocate proper resources to the correct targets based on their demands. The data set is provided by [Figure Eight](https://www.figure-eight.com/) and the messages dealt with are ones that were collected during actual disaster events. Beyond the classification/ML learning problem, this project implements a web app that can take a user message and use the trained model to classify the message. the web app also contains some visualizations generated using the **Flask Engine**.
In terms of sophistication, the focus is made on an ETL pipeline, NLP pipeline and finally an ML pipeline.
1. The ETL Pipeline found in the (**data folder ==> process_data.py file**) cleans the raw messages and generates the proper classification categories. It finally returns an SQLLITE database.
2. The NLP and ML pipeline is contained in the (**models folder ==> train_classifier.py file**) uses the database to extract the necessary data for additional Natural Language processing before training. In terms of feature extraction, we implement the **tfidf transformer**, a custom transformer that extracts some features of interest such as character count, capital word count etc, and another custom transformer that uses **gensym's word2vec** to generate vectors for each sentences in each message. This is done by allowing word2vec to learn the training's set corpus and use the skip gram algorithm to generate vocabulary vectors.
3. Finally, the model is passed through a training pipeline using gridsearch. Two training models are used and tuned. Both implement the multioutput classifier using the random forest classifier in one model and the multinaive bayes estimator in the second model. The model that is used to run the web app is the one obtained from training with all available features using the random forest classifier.



![Screenshot 1](1.PNG)

![Screenshot 2](2.PNG)


# Installation
Make sure any Python 3.* is installed alongside the pandas, numpy, matplotlib, sklearn , pickle, sqlalchemy,NLTK, Plotly and Flask libraries.


# File Descriptions
1. The App folder including the templates folder and "run.py" for the web app.
2. The Data folder containing "Disaster_Data.db", "disaster_categories.csv", "disaster_messages.csv" and "process_data.py".
3. The Models folder including "model_1.sav" and "train_classifier.py" for the NLP pipeline and Machine Learning model.
4. README file
5. Jupyter Notebook that contains all details and steps of the functions, transformers and analysis created.

# Instructions
Run the following in the project's root directory:
**python data/process_data.py** ==> to generate the database
**python models/train_classifier.py** ==> to generate the classifier model that will run the web app. Note: 3 models (model_1, model_2, model_3 are defined and can be used in def main()
**python run.py** ==> to run web app (make sure classifier model name matches the one in run.py)


# Licensing, Authors, Acknowledgements
Thanks to Figure-8 for the data!
Thanks to UDACITY, Kaggle and StackOverflow for providing insight and solution to complications encountered along the way!
