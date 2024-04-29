from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np

def clean(df):
    for column in df.columns:
        df[column] = df[column].str.replace('[', '')
        df[column] = df[column].str.replace(']', '')
        df[column] = df[column].str.replace('\'', '')
        df[column] = df[column].str.replace(' ', '')
        df[column] = df[column].map(lambda x: ', '.join(sorted(x.split(', '))))
    return df

if __name__ == '__main__':
    df_train = pd.read_excel('training_data.xlsx', sheet_name = 'train')
    df_test = pd.read_excel('training_data.xlsx', sheet_name = 'test')
    df_train = clean(df_train)
    df_test = clean(df_test)

    classifiers = [
        KNeighborsClassifier(n_neighbors=4),
        SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None),
        MultinomialNB(),
        LinearSVC(dual='auto'),
        RandomForestClassifier(max_depth=2)
    ]

    for classifier in classifiers:
        text_clf = Pipeline([
            ('vect', TfidfVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', classifier)])
        
        print(' --- ')
        print(text_clf['clf'])

        ##print(text_clf.fit(df_train.Combination, df_train.Leadership).score(df_test.Combination, df_test.Leadership, normalize = True))
        text_clf.fit(df_train.Combination, df_train.Leadership)
        predicted = text_clf.predict(df_test.Combination)
        print(np.mean(predicted == df_test.Leadership))
        
        number_of_attributes = 0
        points = 0
        for i in enumerate(predicted):
            list = i[1].split(',')
            for item in list:
                if item in df_test.Leadership[i[0]]:
                    points += 1
                number_of_attributes += 1
        print(points/number_of_attributes) 
