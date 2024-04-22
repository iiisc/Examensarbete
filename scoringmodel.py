import pandas as pd
import numpy as np
import ast
import os
import re

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.multiclass import OneVsRestClassifier
multilabel = MultiLabelBinarizer()
allCV={}
from PyPDF2 import PdfReader

listOfTitles=[]
myfile=open("stoplista.txt","r",encoding='utf-8')


stoplistVectorizer=TfidfVectorizer(lowercase=True)

stopList=[]
for line in myfile:
   
    data=myfile.readline()
    stopList.append(data)
myfile.close() 

stoplistVectorizer.fit(stopList)
tokenizedStopWords=stoplistVectorizer.get_feature_names_out()

#print(tokenizedStopWords)
tokenizedStopWords=tokenizedStopWords.tolist()





dataframe=pd.read_excel('testfil2.xlsx')
#ast.literal_eval(dataframe['Attribut'].iloc[0])
for i in dataframe['Yrkestitel']:
   listOfTitles.append(dataframe['Yrkestitel'])
print(f"LISTOFTITLES :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::{listOfTitles}")
  



# Rensa och konvertera data till listor av strängar
def clean_and_convert_to_list(text):
    # Kontrollerar först om 'text' är en sträng
    if isinstance(text, str):
        if text.startswith('[') and text.endswith(']'):
            text = text[1:-1]  # Tar bort hakparenteserna
        return [item.strip() for item in text.split(',')]
    else:
        # Returnerar en tom lista om text inte är en sträng
        return []
# Applicera funktionen på Attribut-kolumnen
dataframe['Attribut'] = dataframe['Attribut'].apply(clean_and_convert_to_list)



#ast.literal_eval(dataframe['Attribut'].iloc[0])
#dataframe['Attribut'] = dataframe['Attribut'].apply(lambda x: ast.literal_eval(x))
y=dataframe['Attribut']


#Träning
y = multilabel.fit_transform(dataframe['Attribut'])


print("----------------Bygga model------------------")
dataframe['Attribut'] = dataframe['Attribut'].apply(lambda x: ' '.join(x)) ##Konverterar till en sträng
#tfidf = TfidfVectorizer(analyzer='word', max_features=10000, ngram_range=(1,2), stop_words='english')
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), stop_words=tokenizedStopWords, lowercase=True, )

X = tfidf.fit_transform(dataframe['Yrkestitel'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


def j_score(y_true, y_pred):
  jaccard = np.minimum(y_true, y_pred).sum(axis = 1)/np.maximum(y_true, y_pred).sum(axis = 1)
  return jaccard.mean()*100
def koppladeattributefun(text,attribute):
   pattern=r'\b('+'|'.join(re.escape(attri) for attri in attribute)+r')\b'
   matches =re.findall(pattern,text,flags=re.IGNORECASE)
   return ', '.join(sorted(set(matches),key=matches.index))

while(input("Avsluta? ")!='j'):

  c=input("Välj C ")
  linres=LinearSVC(C=float(c),penalty='l1', dual=False, class_weight="balanced",max_iter=50000)     ##BARA 3 FEL!!!!!!!!
  clf = OneVsRestClassifier(linres)
  clf.fit(X_train, y_train)





  print(multilabel.classes_)


  print("------------------------------Läsa CV--------------------")
  pdfFilePath="./uploads/"
  for files in os.listdir(pdfFilePath):
    numberOfExclusiveHitsForProcent=0

    if files.endswith('.pdf'):
      pdfPath =pdfFilePath+files  
      print(f"Namn på fil {files}")
      #pdf_path = pdf_path+'.pdf'
      wholeDocument=""
      print(pdfPath)
      reader = PdfReader(pdfPath)
      #page = reader.pages[1]
      for sida in reader.pages:
        wholeDocument+=sida.extract_text()+'\n'

    

     # print(f"PDF format i sträng {wholeDocument}")
      CV=wholeDocument
      #CV=page.extract_text()
      #x=[CV]
      x=CV.split()
    #  print(x)
      #print(multilabel.classes_)
      xt=tfidf.transform(x)
      print(f"predicted {clf.predict(xt)}")
      attributesFromCV=multilabel.inverse_transform(clf.predict(xt))
      print(f"attribute :------------------------{attributesFromCV}")

      for i, labels in enumerate(attributesFromCV):
       pass 
        #print(f"input {i}: Med texten:  {x[i]}  ges attributet:")
        #print(f": {' '.join(labels)}'")
         
      


    # CV=page.extract_text()
      wantedAttributes=['affärsmässig','numerisk analytisk förmåga','kvalitetsmedveten','språklig analytisk förmåga']
      #listOfAttributeCleaned= [str(t) for t in attributesFromCV if t]
      listatest=[]
      realCleanList=[]
      score=0
      setOfAttributes=set()
      #listOfAttributeCleaned = [s.strip('(),"') for s in listOfAttributeCleaned]
      listOfAttributeCleaned=attributesFromCV
      print(f"lista med attribute {listOfAttributeCleaned}")
      for cleanAttributes in listOfAttributeCleaned:
         for tuples in cleanAttributes:
              realCleanList.append(tuples)
              setOfAttributes.add(tuples)
      print("---------------------------------------------------------------------------------------------")
      
      print("---------------------------------------------------------------------------------------------")

      print(f"Set of attributes {setOfAttributes}")
      for i in setOfAttributes:
        if i in wantedAttributes:
           print("TRÄFF")
           numberOfExclusiveHitsForProcent=numberOfExclusiveHitsForProcent+1
      for attribut in realCleanList:
         if attribut in wantedAttributes:
            score=score+1

      #print(realCleanList)
 
      
      
      procentOfAttributes=(numberOfExclusiveHitsForProcent/len(wantedAttributes))*100
      print(len(wantedAttributes))
      print(numberOfExclusiveHitsForProcent)
      print(f"Antalet gånger ett attributet uppfylls {wantedAttributes} är {len(realCleanList)}")
      print(f"Andel av attributen som uppfylls {procentOfAttributes} %")
      thisCV={}
      thisCV={"Score": len(realCleanList),"PercentScore":procentOfAttributes}
      allCV.update({files:thisCV})
#print(allCV)
jsonDict=json.dumps(allCV)
print(jsonDict)

