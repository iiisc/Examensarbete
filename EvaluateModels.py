import pandas as pd
import random
from statistics import mean
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

class est_model:
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
    model = est_model
    df_train = pd.read_excel('training_data.xlsx', sheet_name = 'train')
    df_test = pd.read_excel('training_data.xlsx', sheet_name = 'test')
    df_train = model.clean(df_train)
    df_test = model.clean(df_test)

    classifiers = [
        KNeighborsClassifier(),
        LinearSVC(dual='auto'),
        DecisionTreeClassifier(),
        RandomForestClassifier()
    ]
    res = {'Leadership':[], 'Social':[], 'Personal':[], 'Intellectual':[]}
    categories = ['Leadership', 'Social', 'Personal', 'Intellectual']
    classifier_names = ['KNeighborsClassifier', 
                        'LinearSVC',
                        'DecisionTree',
                        'RandomForestClassifier',
                        'Random'
                        ]
    
    j = 0
    for classifier in classifiers:
        for category in categories:
            save_file = classifier_names[j] + "_" + category
            clf = Pipeline([
                ('vect', TfidfVectorizer()),
                ('clf', classifier)])

            clf.fit(df_train.Combination, df_train[category])
            predicted = clf.predict(df_test.Combination)
            number_of_attributes = 0
            points = 0
            for i in enumerate(predicted):
                list = i[1].split(',')
                for item in list:
                    if item in df_test[category][i[0]]:
                        points += 1
                    number_of_attributes += 1
            res[category].append(points/number_of_attributes)
        j += 1
    
    ## Generate result for just guessing attributes:
    all_attributes = {
        'Leadership': ['Ledarskap', 'Tydlig', 'Affärsmässig', 'Strategisk', 'Omdöme', 'Beslutsam', 'Helhetssyn'], 
        'Personal': ['Personlig Mognad', 'Integritet', 'Självständighet', 'Initiativtagande', 'Självgående', 'Flexibel', 'Stabil', 'Prestationsorienterad', 'Energisk', 'Uthållig', 'Mål och resultatorienterad'], 
        'Social': ['Samarbetsförmåga', 'Relationsskapande', 'Empatisk förmåga', 'Muntlig kommunikation', 'Lojal', 'Serviceinriktad', 'Övertygande', 'Kulturell medvetenhet', 'Pedagogisk insikt'], 
        'Intellectual': ['Strukturerad', 'Kvalitetsmedveten', 'Kreativ', 'Specialistkunskap', 'Problemlösande analysförmåga', 'Numerisk analytisk förmåga', 'Språklig analytisk förmåga']}

    random_mean_result = {'Leadership': [], 'Personal': [], 'Social': [], 'Intellectual': []}
    for j in range(1000):
        random_predicted = {'Leadership': [], 'Personal': [], 'Social': [], 'Intellectual': []}
        for row in df_test.T:
            for category in categories:
                sample = ', '.join(random.sample(all_attributes[category], k = 2)).upper()
                random_predicted[category].append(sample)

        for category in categories:
            number_of_attributes = 0
            points = 0
            for i in enumerate(random_predicted[category]):
                list = i[1].split(',')
                for item in list:
                    if item in df_test[category][i[0]]:
                        points += 1
                    number_of_attributes += 1
            random_mean_result[category].append(points/number_of_attributes)
    for category in categories:
        res[category].append(mean(random_mean_result[category]))


    resFrame = pd.DataFrame(res)
    resFrame.insert(0, 'Model', classifier_names)
    print(resFrame.round(decimals=4))

    resFrame.to_excel('Results_Evaluated_Models.xlsx', sheet_name='Evaluated Models', index=False)

    ## Klasserna blir ju alltid kombinationer av 2 attribut. Här får man ut de tre med högst sannolikthet
    ##proba = clf.predict_proba(df_test.Combination)
    ##top_N = np.argsort(proba, axis=1)[:, -3 :]
    ##top_n_with_labels = clf.classes_[top_N]
    ##print(top_n_with_labels)
    ###
