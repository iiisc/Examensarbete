import pandas
from sklearn.feature_extraction.text import CountVectorizer

from PyPDF2 import PdfReader

pdf_path = 'CV.pdf'
reader = PdfReader(pdf_path)
page = reader.pages[0]
listOfTitles=[]
class Category:
    CREATIVE="objekt"




procentOfAquarize=1
choosenSkill="test"
score=0






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

#Lägger till både category och title
for index, col in df_fylld.iterrows():
    if col[0]!=0:
        print("LOOP")
        tempSmallLettersTitle=str(col[0]).lower()
        tempSmallLettersCategory=str(col[1]).lower()
        print("Första  ", tempSmallLettersTitle)
        print("Andra  ",tempSmallLettersCategory)
        #kollar excelfilen och plockar yrkestitel med och parar det med ett personlighetsattribut
        train_x.append(tempSmallLettersTitle)
        temp_y="Category."+tempSmallLettersCategory
        train_y.append(temp_y)
        #FYller lista med titlar
        listOfTitles.append(tempSmallLettersTitle)


vectorizer=CountVectorizer(binary=True, ngram_range=(1,2))
train_x_vectors= vectorizer.fit_transform(train_x)
#print(train_x_vectors.toarray())
from sklearn import svm
clf_svm=svm.SVC(kernel='linear')
#print(clf_svm)
#Tranar och testar datan


testText="jag ar en idrottare"



clf_svm.fit(train_x_vectors, train_y)
test_x=vectorizer.transform([testText])
utskrift=clf_svm.predict(test_x)
print(testText)
print(utskrift)
#print(test_x.toarray())




##FÖRSTA UTKAST PÅ ETT ENKELT POÄNGSYSTEM

""" for i in page.extract_text():
    testText=input("Jobbtitel: ")
    clf_svm.fit(train_x_vectors, train_y)
    test_x=vectorizer.transform([testText])
    utskrift=clf_svm.predict(test_x)
    print(testText)
    print(utskrift)
    if(utskrift==f"Category.{choosenSkill}"):
        print("poäng")
        score=score+1
    #print(test_x.toarray())

print(f"Poäng på attributet {choosenSkill} är {score}") """

##FÖRSTA UTKAST PÅ ETT ENKELT POÄNGSYSTEM som läser in ord från ett PDF och räknar antalet vars kategori stämmer överens
#print(page.extract_text())

CV=page.extract_text()

for i in listOfTitles:
    print(f"Lista av titlar: {i}")

for i in CV.split():
    #print("___________________________________________________________________________")
   
    #print(f"i är : {i}")
    if i.lower() in listOfTitles:
        testText=i
        clf_svm.fit(train_x_vectors, train_y)
        test_x=vectorizer.transform([testText])
        utskrift=clf_svm.predict(test_x)
        print("___________________________________________________________________________")

        print(i)
        print(utskrift)
        if(utskrift==f"Category.{choosenSkill}" ):
            print("poäng")
            score=score+1
        #print(test_x.toarray())

print(f"Poäng på attributet {choosenSkill} är {score}")
