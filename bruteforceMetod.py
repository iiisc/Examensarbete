import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from PyPDF2 import PdfReader

pdf_path = 'CV.pdf'
reader = PdfReader(pdf_path)
page = reader.pages[0]
listOfTitles=[]


procentOfAquarize=1
choosenSkill="test"
score=0

attributesToCheack=input("Skriv in attribute med mellanslag imellan").split()

print(attributesToCheack)


for attribute in attributesToCheack:

    choosenSkill=attribute
    dataframe1= pandas.read_excel(f'{attribute}.xlsx')


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
            print("Title  ", tempSmallLettersTitle)
            print(f"kategori {attribute}")
            #kollar excelfilen och plockar yrkestitel med och parar det med ett personlighetsattribut
            train_x.append(tempSmallLettersTitle)
            temp_y="Category."+attribute.lower()
            train_y.append(temp_y)
            #FYller lista med titlar
            listOfTitles.append(tempSmallLettersTitle)


    vectorizer=CountVectorizer(binary=True, ngram_range=(1,2))
    train_x_vectors= vectorizer.fit_transform(train_x)
    #print(train_x_vectors.toarray())
    from sklearn import svm
    clf_svm=svm.SVC(kernel='linear')
    
    
    # Your text data
    text_data = train_x
    train_x.append("extra")
    train_y.append("Category.extra")
    # Initialize the vectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on your text data and transform the text into numerical vectors
    train_x_vectors = vectorizer.fit_transform(text_data)

    print(f"tränad x {train_x}")
    print(f"tränad y {train_y}")

    ##FÖRSTA UTKAST PÅ ETT ENKELT POÄNGSYSTEM som läser in ord från ett PDF och räknar antalet vars kategori stämmer överens

    CV=page.extract_text()




    for i in listOfTitles:
        print(f"Lista av titlar: {i}")

    for i in CV.split():
        #print("___________________________________________________________________________")
    
        #print(f"i är : {i}")
        if i.lower() in listOfTitles:
            clf_svm.fit(train_x_vectors, train_y)

            test_x=vectorizer.transform([i])
            utskrift=clf_svm.predict(test_x)
            print("___________________________________________________________________________")

            print(i)
            print(utskrift)
            if(utskrift==f"Category.{choosenSkill}" ):
                print("poäng")
                score=score+1
            #print(test_x.toarray())

    print(f"Poäng på attributet {choosenSkill} är {score}")



print(f"Poäng totalt är {score}")
