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
import pandas as pd
#########################################################SKRIVER OM URSPRUNGSFILEN TILL ETT FORMAT SOM PASSAR MODELLEN#########################
def read_and_transform_excel(input_file, output_file):
    # Läser in data från Excel-filen
    df = pd.read_excel(input_file, engine="openpyxl",sheet_name="Attribut satta utifrån kategori")
    
    # Skapar en ny kolumn 'Attribut' som kommer innehålla kommateckenavskilda attribut
    df['Attribut'] = df.apply(lambda row: ', '.join([col.strip().lower() for col in df.columns[1:] if str(row[col]).strip().lower() == 'x']), axis=1)
    print(df.head(30))  # Kontrollera de första 30 raderna
    print(df['Attribut'].value_counts())  # Se fördelningen av hur attribut tilldelas

    # Skapar en ny DataFrame med bara de kolumner vi är intresserade av
    result_df = df[['Yrkestitel', 'Attribut']]
    
    # Sparar den nya DataFrame till en ny Excel-fil
    result_df.to_excel(output_file, index=False)

# Ange sökväg till din input och output Excel-filer
input_file = 'traning.xlsx'
output_file = 'testfil3.xlsx'

# Anropar funktionen med angivna sökvägar
read_and_transform_excel(input_file, output_file)




dataframe=pd.read_excel('testfil.xlsx')
print(dataframe.head())
#print(dataframe['Attribut'].iloc[0])         
#ast.literal_eval(dataframe['Attribut'].iloc[0])

print("-------------------------------- -------------------")





