import pandas as pd
import numpy as np
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
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
    dataframe = pd.read_excel('training_data.xlsx')
    for i in dataframe['Combination']:
      self.listOfTitles.append(dataframe['Combination'])
    dataframe['Leadership'] = (dataframe['Leadership'].apply(self.clean_and_convert_to_list))
    #dataframe['Max']= dataframe['Max'].to_list()
    #print( dataframe['Max'])
    y = self.multilabel.fit_transform(dataframe['Leadership'])
    #print(f"multilabel är {self.multilabel.classes_}")
    dataframe['Leadership'] = dataframe['Leadership'].apply(lambda x: ' '.join(x)) ##Konverterar till en sträng
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), stop_words= self.stopList(), lowercase=True, )
    X = tfidf.fit_transform(dataframe['Combination'])
    #print("VOC: ", tfidf.vocabulary_)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 
      ###TRÄNA MODELL
      ###SPARA MODEL OCH DATA
    if 1==1:
      #linres=LinearSVC(C=1.5,penalty='l1', dual=False, class_weight="balanced",max_iter=50000)
      linres=SVC(C=1.5, class_weight="balanced",max_iter=50000, probability=True)
      clf = OneVsRestClassifier(linres)
      pickle.dump(clf,open(modelName,'wb+'))
    else:
      clf=pickle.load(open(modelName, 'rb+'))
  
    clf.fit(X_train, y_train)

    print(f"Score är {clf.score(X_train, y_train)}")
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
    testframe= pd.read_excel('training_data.xlsx', 'test')
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
      
      print(x)
      #print(f"listan litet x {x}")
      roundCounter=0
      #####---TEST SCORE-----#####
      totalScore=0
      for index,titlar in enumerate(testframe['Combination']):
         roundScore=0
         roundCounter=roundCounter+1

         attributesFromCV=[]
         a=[]
         a.append(titlar)
         predictedList=[]
         xt=self.tfidf.transform(a)
         
        
        
         print(f"proba {self.clf.predict_proba(xt)}")
         lista=self.clf.predict_proba(xt)
         print(lista)
         print(f"ax är {self.clf.predict(xt)}")
        #
         print(f"multilabel {self.multilabel.classes_}")

         attributesFromCV=( self.multilabel.inverse_transform(self.clf.predict(xt)))
         for att in attributesFromCV[0]:
            predictedList.append(att.strip('"()').replace("'", ""))
                ######BEHANLDAR FÖRUTSPÅTT

        # print(f"predictedLista är {predictedList}")
     
        ######BEHANLDAR FACIT
        # print(f"För titlarna {titlar} ges attributen: {attributesFromCV}")
         tempList2=[]
         tempList2.append(testframe['Leadership'][index])
         facitString=testframe['Leadership'][index]
         facitString=facitString.strip("'[],")
         facitString=facitString.replace("'","").lower()
         facitString=facitString.replace(",","").lower()
         facitStringLista=[]

         facitStringLista=facitString.split(" ")
         facitStringLista.sort()
         predictedList.sort()
         print(f"Jämför {predictedList} med ")
         print(f"med    {facitStringLista}  ")

         if facitStringLista==predictedList:
            print("LIIIIIIKKKKAAA")
            totalScore=totalScore+1
            print("-----------------------------------------------------------------------------------------------------------")

         else:
              
          for attribut in predictedList:
      #      print(f"attributet {attribut}")
            for facit in facitStringLista:
      #        print(f"facit är {facit}")
              if facit==attribut:  
         #       print(f"attribute {attribut} i {testframe['Leadership'][index]}" )
                roundScore=roundScore+1   
        #        print("träff")
          if roundScore>=len(facitStringLista)/2:
            totalScore=totalScore+1
            print("totalscore")
          print("-----------------------------------------------------------------------------------------------------------")
      
      
      
      
      
      
      print(f"TOTAL POÄNG PÅ MODELN ÄR {totalScore} vilket ger en procent på: {(totalScore/roundCounter)*100} utav totalt {roundCounter} test")
      
      
      
      xt = self.tfidf.transform(x)
      attributesFromCV = self.multilabel.inverse_transform(self.clf.predict(xt))
      realCleanList=[]
      listOfAttributeCleaned=attributesFromCV
      for cleanAttributes in listOfAttributeCleaned:
        for tuples in cleanAttributes:
              realCleanList.append(tuples)
              #etOfAttributes.add(tuples)
              
      scoringDict = {}
      for attribut in realCleanList:
        scoringDict[attribut] = scoringDict.get(attribut, 0) + 1

    print(f"lista med förutspådda attribut: {attributesFromCV}")
    return returnDict

if __name__ == '__main__':
  model = Model()
  CV="Bagare Tolk chef Bagare', 'Tolk', 'Brandman' Bagare Ekonom, chef 'Bagare', 'Ekonom', 'Brandman' 'Bagare', 'Chef', 'Brandman "
  model.train_model()
  print(model.run_model(CV))