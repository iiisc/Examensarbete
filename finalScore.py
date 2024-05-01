from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import RandomizedSearchCV

import pandas as pd
import numpy as np
class carl_model:
    def __init__(self):
        self.df_train = None
        self.df_test = None

    def clean(df):
        for column in df.columns:
            df[column] = df[column].str.replace('[', '')
            df[column] = df[column].str.replace(']', '')
            df[column] = df[column].str.replace('\'', '')
            df[column] = df[column].map(lambda x: ', '.join(sorted(x.split(', '))))
        return df
    
    def readExcel(self):
        df_train = pd.read_excel('training_data.xlsx', sheet_name = 'train')
        df_test = pd.read_excel('training_data.xlsx', sheet_name = 'test')
        self.df_train = self.clean(df_train)
        self.df_test = self.clean(df_test)

if __name__ == '__main__':
    model = carl_model
    df_train = pd.read_excel('training_data.xlsx', sheet_name = 'train')
    df_test = pd.read_excel('training_data.xlsx', sheet_name = 'test')
    df_train = model.clean(df_train)
    df_test = model.clean(df_test)

  
    
    res = {'Leadership':[], 'Social':[], 'Personal':[], 'Intellectual':[]}
    categories = ['Leadership', 'Social', 'Personal', 'Intellectual']
    classifier=LinearSVC(dual=True)
    tfid=TfidfVectorizer()
    clf = Pipeline([
                ('vect', TfidfVectorizer(analyzer='word')),
                ('tfidf', TfidfTransformer()),
                ('clf', classifier)])
    for category in categories:
            print(f"kategori: ", category)

            clf.fit(df_train.Combination, df_train[category]),
            predicted = clf.predict(df_test.Combination)

            testpredict=["LÄKARE, POLIS, LÄRARE"]
            print(f"attribut för cv: {clf.predict(testpredict)}")
            number_of_attributes = 0
            points = 0
            for i in enumerate(predicted):
                list = i[1].split(',')
                for item in list:
                    if item in df_test[category][i[0]]:
                        points += 1
                    number_of_attributes += 1
            res[category].append(points/number_of_attributes)
            ##print("np.mean score: ", np.mean(predicted == df_test[category]))

    resFrame = pd.DataFrame(res)
    resFrame.insert(0, 'Model', classifier)
    print(resFrame)