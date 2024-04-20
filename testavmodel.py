import pandas as pd
import numpy as np
import ast

from fuzzywuzzy import process
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.multiclass import OneVsRestClassifier
multilabel = MultiLabelBinarizer()

dataframe=pd.read_excel('testfil2.xlsx')
print(dataframe.head())
#print(dataframe['Attribut'].iloc[0])         
#ast.literal_eval(dataframe['Attribut'].iloc[0])
print(dataframe['Attribut'].iloc[0])

print("y= -------------------")
print(type(dataframe['Attribut'].iloc[0]))


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
print(y)
print("---------Träning---------")
#Träning
y = multilabel.fit_transform(dataframe['Attribut'])
print(y)

print("-----------multilabel classer________________________------")
#print(multilabel.classes_)

print("---------------------uppradat------------------------")
print(pd.DataFrame(y,columns=multilabel.classes_))
listOfTitlesAfterProcessing=[]

for i in dataframe['Yrkestitel']:
   listOfTitlesAfterProcessing.append(i.replace("inom", ""))
myfile=open("stoplista.txt","r",encoding='utf-8')


print("--------------------------------------------------------SKAPA STOPLISTA---------------------------------------------------------------------------------------------------------")

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


print(f"Type av dataframe {type(dataframe['Yrkestitel'])}")
print("--------------------------------------------------------Bygga model---------------------------------------------------------------------")
dataframe['Attribut'] = dataframe['Attribut'].apply(lambda x: ' '.join(x)) ##Konverterar till en sträng
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), stop_words=tokenizedStopWords, lowercase=True, )
X = tfidf.fit_transform(dataframe['Yrkestitel'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
for words in tfidf.vocabulary_:
    print(f"valda ord för tokenizer {words}") 
print(f"Längd på lista av vocabluary {len(tfidf.vocabulary_)}")
sgd = SGDClassifier()
lr = LogisticRegression(solver='liblinear-')
svc = LinearSVC()

def j_score(y_true, y_pred):
  jaccard = np.minimum(y_true, y_pred).sum(axis = 1)/np.maximum(y_true, y_pred).sum(axis = 1)
  return jaccard.mean()*100


def print_score(y_pred, clf):
  print("Clf: ", clf.__class__.__name__)
  print('Jacard score: {}'.format(j_score(y_test, y_pred)))
  print('----')

""" for classifier in [LinearSVC(C=999999,penalty='l2', dual=False, multi_class="ovr", verbose=10, fit_intercept=False)]:
  clf = OneVsRestClassifier(classifier)
  LogisticRegression(solver='l2', dual=False,C=10000)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print_score(y_pred, classifier) """

  
solvers = ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga']
""" for solver in solvers:
    clf = OneVsRestClassifier(LogisticRegression(solver=solver, max_iter=1000))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Solver: {solver}, Accuracy: {accuracy_score(y_test, y_pred)}")

 """


classifiers = [
    LogisticRegression(max_iter=1000),
    LinearSVC(),
    RandomForestClassifier()
]




""" for classifier in [LinearSVC(C=99999999999999999999,penalty='l2', dual=False, tol=4)]:
  clf = OneVsRestClassifier(classifier)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print_score(y_pred, classifier)  
 """

####################################################STÄLL IN OLIKA PARAMETRAR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!##########################################
linres=LinearSVC(C=99999999999999999999,penalty='l2', dual=False )
logres=LogisticRegression()

linres2=LinearSVC(C=99999999999999999999,penalty='l1', dual=False)
logres2=LogisticRegression(solver="liblinear",multi_class='ovr',penalty='l2',C=1.2)

linres3=LinearSVC(C=99999999999999999999,penalty='l1', dual=False, class_weight="balanced"  )     ##BARA 3 FEL!!!!!!!!
logres3=LogisticRegression(solver="newton-cg",penalty='l2',C=10)

linres4=LinearSVC(C=99999999999999999999,penalty='l1', dual=False, class_weight="balanced" ,multi_class="ovr" ) ##BARA 3 FEL!!!!!!!!
logres4=LogisticRegression(solver="newton-cholesky",multi_class='ovr',C=100,class_weight="balanced",penalty="l2")

linres5=LinearSVC(C=99999999999999999999,penalty='l1', dual=False, class_weight="balanced" ,multi_class="crammer_singer")
logres5=LogisticRegression(solver="liblinear",multi_class='ovr',C=100,class_weight="balanced",dual=False)

#randfor=RandomForestClassifier()
####LÄGG IN NYA NAMN I LISTAN SÅ TESTAS DESSA
#classifiers=[linres,logres,linres2,logres2,linres3,logres3,linres4,logres4,linres5,logres5]
classifiers=[linres4]


print("----------------------------testa model----------------------")

antaltitlar=0
counter=0
bestScores={}
round=0
roundcounter=0
for classifier in classifiers:
  
  if roundcounter%2==0:
    round=round+1

    print(f"-------------------------------------------RUNDA {round} ------------------------------------------------------")
  roundcounter=roundcounter+1
  

  clf=OneVsRestClassifier(classifier)
  clf.fit(X_train, y_train)
  antaltitlar=0
  counter=0
  for titlar in dataframe['Yrkestitel']:
      antaltitlar=antaltitlar+1

      x=[titlar]

      #print(x)
      #print(multilabel.classes_)
      xt=tfidf.transform(x)
      #print(clf.predict(xt))
      temp=multilabel.inverse_transform(clf.predict(xt))

      #print(multilabel.inverse_transform(clf.predict(xt)))
      if len(temp[0])==0:
        counter=counter+1
        #print(f"Har inte kopplats: {titlar}")
      else:
        pass
        ###För att kontrollera vilket jobb som kopplats till vad avkommentera nedanståend två rader
        #print(f"Har kopplats: {titlar} till:")
        #print(f"{temp}\n")
      
      
      #print("Utskrift------------",clf.predict(xt))
      #print("_____________________________________________________________________________________________________________________________________")
  print(f"Test av classifire {str(classifier)}")
  print(f"Antal fel {counter} i procent {(counter/antaltitlar)*100} totalt finns det {antaltitlar} titlar")
  print("")
  bestScores[classifier]=counter
#print(bestScores)


x=['Systemutvecklare']


print(f"Antal fel {counter} i procent {(counter/antaltitlar)*100} totalt finns det {antaltitlar} titlar")
x=['Systemutvecklare']

#print(x)
#print(multilabel.classes_)
xt=tfidf.transform(x)
#print(clf.predict(xt))
temp=multilabel.inverse_transform(clf.predict(xt))
print(temp)

""" for classifier in classifiers:
    clf = OneVsRestClassifier(classifier)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Classifier: {classifier.__class__.__name__}, Accuracy: {accuracy_score(y_test, y_pred)}") """
""" print("-------------------------------------debugg------------------------")
print("Etikettklasser:", multilabel.classes_)
target_label = multilabel.classes_[4]
print("Etikett för index 4:", target_label)
dataframe['has_target_label'] = np.array(y)[:, 4] == 1  # Index 4 är den etikett vi undersöker
filtered_examples = dataframe[dataframe['has_target_label']]
print("Exempel med etiketten '{}':".format(target_label))
print(filtered_examples) """






