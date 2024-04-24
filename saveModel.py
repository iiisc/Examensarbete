import pandas as pd
import numpy as np
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from PyPDF2 import PdfReader
import pickle

class Model:
  def __init__(self):
    self.allCV={}
    self.multilabel = MultiLabelBinarizer()
    self.listOfTitles=[]
    self.tfidf = ""
    self.clf = ""

  def openLists():
    textList=[]
    kategori1 = open("kategori1.txt","r",encoding='utf-8')
    for line in kategori1:
        textList.append(line.strip())
    kategori1.close()
    return textList

  def stopList(self):
    myfile=open("stoplista.txt","r",encoding='utf-8')
    stoplistVectorizer=TfidfVectorizer(lowercase=True)
    stopList=[]
    for line in myfile:
        stopList.append(line)
    myfile.close() 

    stoplistVectorizer.fit(stopList)
    tokenizedStopWords=stoplistVectorizer.get_feature_names_out()
    tokenizedStopWords=tokenizedStopWords.tolist()
    return tokenizedStopWords

  # Rensa och konvertera data till listor av strängar
  def clean_and_convert_to_list(self, text):
      # Kontrollerar först om 'text' är en sträng
      if isinstance(text, str):
          if text.startswith('[') and text.endswith(']'):
              text = text[1:-1]  # Tar bort hakparenteserna
          return [item.strip() for item in text.split(',')]
      else:
          # Returnerar en tom lista om text inte är en sträng
          return []
  # Applicera funktionen på Attribut-kolumnen

  def readPDFCV(self, fileName: str, pdfFilePath):
    print("------------------------------Läsa CV--------------------")
    wholeDocument=""
    if fileName.endswith('.pdf'):
        pdfPath =pdfFilePath+fileName  
        print(f"Namn på fil {fileName}")
        print(pdfPath)
        reader = PdfReader(pdfPath)
        for sida in reader.pages:
          wholeDocument+=sida.extract_text()+'\n'
    return wholeDocument
  
  def cheackForChanges(self):
      traningData="testfil2.xlsx"
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
      

  ###VECTORIZE/TOKENIZ
  def train_model(self):
    print("----------------Bygga model------------------")
    modelName="model.sav"
    tokenDataName="token.pkl"
    trainTestData="traintestdata.pkl"
    tifdfData="tifdf.pkl"
    if self.cheackForChanges():
    ###ÖPPNA EXCELDOKUMENT
      dataframe = pd.read_excel('testfil2.xlsx')
      for i in dataframe['Yrkestitel']:
        self.listOfTitles.append(dataframe['Yrkestitel'])
      dataframe['Attribut'] = (dataframe['Attribut'].apply(self.clean_and_convert_to_list))
      
      y = self.multilabel.fit_transform(dataframe['Attribut'])
      dataframe['Attribut'] = dataframe['Attribut'].apply(lambda x: ' '.join(x)) ##Konverterar till en sträng
      tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), stop_words= self.stopList(), lowercase=True, )

      X = tfidf.fit_transform(dataframe['Yrkestitel'])
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 
      ###TRÄNA MODELL
      linres=LinearSVC(C=2,penalty='l1', dual=False, class_weight="balanced",max_iter=50000)
      clf = OneVsRestClassifier(linres)
      clf.fit(X_train, y_train)

      ###SPARA MODEL OCH DATA
      pickle.dump(X,open(tokenDataName,'wb') )
      pickle.dump(clf,open(modelName,'wb'))
     
      with open('X_train.pkl', 'wb') as file:
          pickle.dump(X_train, file)
      with open('X_test.pkl', 'wb') as file:
          pickle.dump(X_test, file)
      with open('y_train.pkl', 'wb') as file:
          pickle.dump(y_train, file)
      with open('y_test.pkl', 'wb') as file:
          pickle.dump(y_test, file)
     # with open(trainTestData,'wb') as data:
     #    pickle.dump((X_train, X_test, y_train, y_test),data)
      with open (tifdfData,'wb') as data:
         pickle.dump(tfidf,data)
      print("Tränad")
    else:
      X=pickle.load(open(tokenDataName,'rb'))
      clf=pickle.load(open(modelName, 'rb'))
      #ith open(trainTestData,'rb') as data:
       # X_train, X_test, y_train, y_test=pickle.load(data)
      with open('X_train.pkl', 'rb') as file:
          loaded_X_train = pickle.load(file)
      with open('X_test.pkl', 'rb') as file:
          loaded_X_test = pickle.load(file)
      with open('y_train.pkl', 'rb') as file:
          loaded_y_train = pickle.load(file)
      with open('y_test.pkl', 'rb') as file:
          loaded_y_test = pickle.load(file)
      
      
      with open(tifdfData,'rb') as data:
        tfidf=pickle.load(data)
      print("laddad")
  
      clf.fit(loaded_X_train, loaded_y_train)
    self.tfidf = tfidf
    self.clf = clf
    return

  def j_score(y_true, y_pred):
    jaccard = np.minimum(y_true, y_pred).sum(axis = 1)/np.maximum(y_true, y_pred).sum(axis = 1)
    return jaccard.mean()*100

  def run_model(self, fileList:list):
    returnDict = {}
    pdfFilePath="./uploads/"
    for files in fileList:
      CV:str = self.readPDFCV(files, pdfFilePath)
      x = CV.split()
      xt = self.tfidf.transform(x)
      attributesFromCV = self.multilabel.inverse_transform(self.clf.predict(xt))
      realCleanList=[]
      setOfAttributes=set()
      listOfAttributeCleaned=attributesFromCV
      for cleanAttributes in listOfAttributeCleaned:
        for tuples in cleanAttributes:
              realCleanList.append(tuples)
              setOfAttributes.add(tuples)
              
      scoringDict = {}
      for attribut in realCleanList:
        scoringDict[attribut] = scoringDict.get(attribut, 0) + 1

      returnDict.update({files: scoringDict})
    return returnDict

if __name__ == '__main__':
  model = Model()
  model.train_model()
  print(model.run_model(["CV.pdf"]))