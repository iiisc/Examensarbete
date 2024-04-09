import pandas as pd

import openpyxl





listOfTitles=["jobb1","jobb2","jobb3"]


listOfMainCategories=["main1","main2"]

data=[]

def main1():
    counter=1
    for i in(listOfCategoriesTitle1):
        print(f"\n{counter}: {i}")
        counter=counter+1

    choosen=input("Välj attribute: ")
    return int(choosen)
def main2():
    counter=1
    for i in(listOfCategoriesTitle2):
        print(f"\n{counter}: {i}")
        counter=counter+1 
    choosen=input("Välj attribute: ")
    return int(choosen)
def main3():
    counter=1
    for i in(listOfCategoriesTitle3):
        print(f"\n{counter}: {i}")
        counter=counter+1 
    choosen=input("Välj attribute: ")
    return int(choosen)
def main4():
    counter=1
    for i in(listOfCategoriesTitle4):
        print(f"\n{counter}: {i}")
        counter=counter+1 
    choosen=input("Välj attribute: ")
    return int(choosen)





def read():
    dokumentnamn=input("Namn på textdokument: ")
    listOfTitle=[]
    with open(f'{dokumentnamn}.txt','r') as file:
        for i in file:
            print(i.strip())
            listOfTitle.append(i.strip())

    return listOfTitle


listOfTitles=read()
listOfCategoriesTitle1=read()
listOfCategoriesTitle2=read()
listOfCategoriesTitle3=read()
listOfCategoriesTitle4=read()

for i in listOfTitles:
    print(f"\nTitle är {i}")
    match input("\nVälj huvudkategori: \nPersonal Abilities 1 \nSocial Abilities 2\nLeadership Abilities \nIntellectual Abilities\n _______________________________________\n"):
        case '1':
            choosen=main1()
            inputData=[f'{i}',f'{listOfCategoriesTitle1[choosen-1]}']
            data.append(inputData)

        case '2':        
            choosen=main2()
            inputData=[f'{i}',f'{listOfCategoriesTitle2[choosen-1]}']
            data.append(inputData)
        case '3':
            choosen=main3()
            inputData=[f'{i}',f'{listOfCategoriesTitle2[choosen-1]}']
            data.append(inputData)
        case '4':
            choosen=main4()
            inputData=[f'{i}',f'{listOfCategoriesTitle2[choosen-1]}']
            data.append(inputData)


df = pd.DataFrame(data,columns=['Title', 'Category'])

print(df)
namn=input("Namn på dokument ")
df.to_excel(namn+".xlsx",sheet_name='Blad ett' )