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
      text=text.lower()
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
        reader = PdfReader(pdfPath)
        for sida in reader.pages:
          wholeDocument+=sida.extract_text()+'\n'
    return wholeDocument
  def cheackForChanges(self):
      traningData="testfleratitlar.xlsx"
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
    modelName="model2.sav"

    ###ÖPPNA EXCELDOKUMENT
    dataframe = pd.read_excel('carl_test_res.xlsx')
    for i in dataframe['Combination']:
      self.listOfTitles.append(dataframe['Combination'])
    dataframe['Max'] = (dataframe['Max'].apply(self.clean_and_convert_to_list))
    #dataframe['Max']= dataframe['Max'].to_list()
    #print( dataframe['Max'])
    y = self.multilabel.fit_transform(dataframe['Max'])
    #print(f"multilabel är {self.multilabel.classes_}")
    dataframe['Max'] = dataframe['Max'].apply(lambda x: ' '.join(x)) ##Konverterar till en sträng
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(2,3), stop_words= self.stopList(), lowercase=True, )
    X = tfidf.fit_transform(dataframe['Combination'])
    #print("VOC: ", tfidf.vocabulary_)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 
      ###TRÄNA MODELL
      ###SPARA MODEL OCH DATA
    
    if 1==1:
      linres=LinearSVC(C=1.1,penalty='l1', dual=False, class_weight="balanced",max_iter=50000)
      clf = OneVsRestClassifier(linres)
      pickle.dump(clf,open(modelName,'wb+'))
    else:
      clf=pickle.load(open(modelName, 'rb+'))
  
    clf.fit(X_train, y_train)

    self.tfidf = tfidf
    self.clf = clf
    return

  def j_score(y_true, y_pred):
    jaccard = np.minimum(y_true, y_pred).sum(axis = 1)/np.maximum(y_true, y_pred).sum(axis = 1)
    return jaccard.mean()*100

  def run_model(self, fileList:str):
    listWithTitles=["Polis","Brandman","Sjuksköterska","Läkare","Pilot","Lärare","Bagare","Systemutvecklare","Ekonom","Chef"]
    returnDict = {}
    pdfFilePath="./uploads/"
    for i in range(1):
      #CV:str = self.readPDFCV(files, pdfFilePath)
      #CV=fileList.split()
      x=[]
      x.append(fileList)
      #for words in CV:
        #print(f"Ord i CV: {words}")
       # if words in listWithTitles:
       #     x.append(words)
            #print(f"Jobbtitlet förhoppningsvis: {words}")
      
    
      #print(f"listan litet x {x}")
      for titlar in x:
         a=[]
         a.append(titlar)
         xt=self.tfidf.transform(a)
         attributesFromCV = self.multilabel.inverse_transform(self.clf.predict(xt))

         print(f"För titlarna {titlar} ges attributen: {attributesFromCV}")
         print("-----------------------------------------------------------------------------------------------------------")
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

    print(f"lista med förutspådda attribut: {attributesFromCV}")

    return returnDict

if __name__ == '__main__':
  model = Model()
  CV="Bagare Tolk chef ","['Bagare', 'Tolk', 'Brandman']","Bagare Ekonom, chef","['Bagare', 'Ekonom', 'Brandman']","['Bagare', 'Chef', 'Brandman']"
  model.train_model()
  print(model.run_model(CV))