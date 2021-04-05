import sys
# import libraries
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine
import nltk
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, fbeta_score, make_scorer, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def load_data(database_filepath):
    '''
    Function for loading the database into pandas DataFrames
    Args: database_filepath: the path of the database
    Returns:    X: features (messages)
                y: categories (one-hot encoded)
                An ordered list of categories
    '''
    
    # load data from database
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql('SELECT * FROM disaster_response', engine)
    print(df.columns.values)
    # Define feature and target variables X and Y
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)     
    #Y['related'] = Y['related'].map(lambda x: 1 if x==2 else x)
    categories = Y.columns.tolist()
    
    return X, Y, categories

def tokenize(text):
    '''
    Function for tokenizing string
    Args: Text string
    Returns: List of tokens
    '''    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    stop_words = stopwords.words("english")

    tokens = word_tokenize(text)
    
    lemmatizer = nltk.WordNetLemmatizer()
    
    return [lemmatizer.lemmatize(token).lower().strip() for token in tokens if token not in stop_words]
 
def build_model():
    '''
    Function for building pipeline and GridSearch
    Args: None
    Returns: Model (GridSearchCV)
    '''
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100), n_jobs=-1))
                        ])

    # model parameters for GridSerchcv
    parameters = {
        'vect__max_df': [0.8, 1.0],
        'clf__estimator__n_estimators': [10, 20],
        'clf__estimator__min_samples_split': [2, 5]    
    }

    ftwo_scorer = make_scorer(fbeta_score, beta=2)
    # Initialize a gridsearchcv object that is parallelized
    #cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', verbose=1)
    #cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1', cv=3, verbose=10, n_jobs=-1)
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=1, n_jobs=-1)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    # Generate predictions    
    Y_pred = model.predict(X_test)

    # Print out the full classification report
    print(classification_report(Y_test, Y_pred, target_names=category_names))

    #for  idx, cat in enumerate(Y_test.columns.values):
    #    print("{} -- {}".format(cat, accuracy_score(Y_test.values[:,idx], y_pred[:, idx])))
    #print("accuracy = {}".format(accuracy_score(Y_test, y_pred)))    

def save_model(model, model_filepath):
    '''
    Function for saving the model as picklefile
    Args: Model, filepath
    Returns: Nothing. Saves model to pickle file
    '''
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)


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