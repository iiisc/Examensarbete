import pandas
from sklearn.feature_extraction.text import CountVectorizer

class Category:
    CREATIVE="Kreativ"
    METICULOUS="Noggrann"



dataframe1= pandas.read_excel('traning.xlsx')


print(type(dataframe1))
print(dataframe1)

#Tar bort alla tomma rader
dataframe2=dataframe1.dropna()

#fyller alla tomma rader och sätter dit en 0
df_fylld = dataframe1.fillna(0)
train_x=[]
train_y=[]
#skriver ut rad för rad

for index, col in df_fylld.iterrows():
    if col[0]!=0:
        print("LOOP")
        print("Första  ",col[0])
        print("Andra  ",col[1])
        #kollar excelfilen och plockar yrkestitel med och parar det med ett personlighetsattribut
        train_x.append(col[0])
        temp_y="Category."+str(col[1])
        train_y.append(temp_y)


vectorizer=CountVectorizer(binary=True, ngram_range=(1,2))
train_x_vectors= vectorizer.fit_transform(train_x)
#print(train_x_vectors.toarray())
from sklearn import svm
clf_svm=svm.SVC(kernel='linear')
#print(clf_svm)
#Tranar och testar datan
testText="jag ar en konstnar"
clf_svm.fit(train_x_vectors, train_y)
test_x=vectorizer.transform([testText])
utskrift=clf_svm.predict(test_x)
print(testText)
print(utskrift)
#print(test_x.toarray())
