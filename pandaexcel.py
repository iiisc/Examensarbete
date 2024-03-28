
import pandas
dataframe1= pandas.read_excel('bok1.xlsx')


print(type(dataframe1))
print(dataframe1)

#Tar bort alla tomma rader
dataframe2=dataframe1.dropna()

#fyller alla tomma rader och sätter dit en 0
df_fylld = dataframe1.fillna(0)
#skriver ut rad för rad
for index, row in df_fylld.iterrows():
    if row[2]!=0:
        print(row[2])