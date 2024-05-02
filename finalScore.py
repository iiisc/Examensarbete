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
from PyPDF2 import PdfReader
import pickle
import json
import os

import pandas as pd
import numpy as np
class model:
    def __init__(self, filesToRead:list):
        self.df_train = self.clean(pd.read_excel('training_data.xlsx', sheet_name = 'train'))
        self.df_test = self.clean(pd.read_excel('training_data.xlsx', sheet_name = 'test'))
        self.modelName="model3.sav"
        self.filelist = filesToRead
        self.toPredict = []

        if self.cheackForChanges():
            self.createModel()
            self.saveModel()
            print("Skapar model")
        else:
            self.clf=pickle.load(open(self.modelName, 'rb'))
            print("Laddar model")

        self.res = {'Leadership':[], 'Social':[], 'Personal':[], 'Intellectual':[]}
        self.categories = ['Leadership', 'Social', 'Personal', 'Intellectual']
        self.readFiles()
        self.predictAttributes()

    def clean(self,df):
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

    def saveModel(self):
      pickle.dump(self.clf,open(self.modelName,'wb'))

    def readPDFCV(fileName: str):
        print("------------------------------Läsa CV--------------------")
        wholeDocument=""
        pdfFilePath="Uploads"
        if fileName.endswith('.pdf'):
            pdfPath = os.path.join(pdfFilePath, fileName)
            print("READPDFCV: ", pdfPath)
            reader = PdfReader(pdfPath)
            for sida in reader.pages:
                wholeDocument+=sida.extract_text()+'\n'
        return wholeDocument
    
    def cheackForChanges(self):
        traningData="training_data.xlsx"
        filePath="./"+traningData
        metaData="./metaData.json"

        excelTraningDataProperties=os.stat(filePath)
        jsonExcelProperties=json.dumps(excelTraningDataProperties.st_mtime)
        with open(metaData,'r') as data:
            oldProperties=data.read()
        if oldProperties!=str(jsonExcelProperties):
            with open(metaData, 'w') as outData:
                outData.write(jsonExcelProperties)
            return True
        return False
    
    def createModel(self):
        self.classifier=LinearSVC(dual=True)
        self.tfid=TfidfVectorizer()
        self.clf = Pipeline([
                ('vect', TfidfVectorizer(analyzer='word')),
                ('tfidf', TfidfTransformer()),
                ('clf', self.classifier)])
        #self.clf.fit(self.df_train.Combination, self.df_train[category]),

    def predictAttributes(self):
        predictDict={"Name:" :self.filelist}
        for category in self.categories:
            self.clf.fit(self.df_train.Combination, self.df_train[category]),
            predicted = self.clf.predict(self.toPredict)
            predictDict[category]=predicted
        returnFrame = pd.DataFrame(predictDict)
        return returnFrame

    def readFiles(self):
        print('self.filelist: ', self.filelist)
        df=pd.read_excel("carl_test.xlsx")
        listOfJobTitles=df.Yrkestitel.to_list()
        for file in self.filelist:
            listOfJobTitlesFromCV=""
            CVread=[model.readPDFCV(file)]
            for word in CVread[0].split():
                if word in listOfJobTitles:
                    listOfJobTitlesFromCV=listOfJobTitlesFromCV+word+" "
            print(f"lista av jobtitlar {listOfJobTitlesFromCV}")
            self.toPredict.append(listOfJobTitlesFromCV)
        return

if __name__ == '__main__':
    model = model(['läkaresjuksköterska.pdf', 'polisbrandman.pdf', 'CV.pdf'])
    print(model.predictAttributes())