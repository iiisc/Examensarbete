import pandas as pd
import numpy as np
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

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

stopListVectorizer=TfidfVectorizer(lowercase=True)
myfile=open("stoplista.txt","r",encoding='utf-8')


stopList=[]
for line in myfile:
   
    data=myfile.readline()
    stopList.append(data)
myfile.close() 

stopListVectorizer.fit(stopList)
tokenizedStopWords=stopListVectorizer.get_feature_names_out()

#print(tokenizedStopWords)
tokenizedStopWords=tokenizedStopWords.tolist()





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
print(multilabel.classes_)

print("---------------------uppradat------------------------")
print(pd.DataFrame(y,columns=multilabel.classes_))
print(tokenizedStopWords)
print("----------------Bygga model------------------")
dataframe['Attribut'] = dataframe['Attribut'].apply(lambda x: ' '.join(x)) ##Konverterar till en sträng
tfidf = TfidfVectorizer(analyzer='word', max_features=10000, ngram_range=(1,2), stop_words=tokenizedStopWords)
X = tfidf.fit_transform(dataframe['Yrkestitel'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train,y_test= tr
sgd = SGDClassifier()
lr = LogisticRegression(solver='lbfgs')
svc = LinearSVC()
def j_score(y_true, y_pred):
  jaccard = np.minimum(y_true, y_pred).sum(axis = 1)/np.maximum(y_true, y_pred).sum(axis = 1)
  return jaccard.mean()*100


def print_score(y_pred, clf):
  print("Clf: ", clf.__class__.__name__)
  print('Jacard score: {}'.format(j_score(y_test, y_pred)))
  print('----')

for classifier in [LinearSVC(C=1.5, penalty = 'l1', dual=False)]:
  clf = OneVsRestClassifier(classifier)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print_score(y_pred, classifier)

for classifier in [LinearSVC(C=1.5, penalty = 'l1', dual=False)]:
  clf = OneVsRestClassifier(classifier)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print_score(y_pred, classifier)  



print("----------------------------testa model----------------------")




while(1):
  x=[]
  x.append(input("Testa ord: "))
  print(x)
  #print(multilabel.classes_)
  xt=tfidf.transform(x)
  print(clf.predict(xt))

  print(multilabel.inverse_transform(clf.predict(xt)))


  print("Utskrift------------",clf.predict(xt))

  ordstrang=multilabel.inverse_transform(clf.predict(xt))
  ordstrang=ordstrang[0]
  print(type(ordstrang))
  print(ordstrang)
  for i in ordstrang:
     print(i)





""" print("-------------------------------------debugg------------------------")
print("Etikettklasser:", multilabel.classes_)
target_label = multilabel.classes_[4]
print("Etikett för index 4:", target_label)
dataframe['has_target_label'] = np.array(y)[:, 4] == 1  # Index 4 är den etikett vi undersöker
filtered_examples = dataframe[dataframe['has_target_label']]
print("Exempel med etiketten '{}':".format(target_label))
print(filtered_examples) """






