import pandas as pd
from itertools import combinations

def createNewRowFromIndex(df):
    """Takes pandas dataframe with multiple indexes and returns summarized row"""
    df = df.T
    newRow = {}
    for (columnName, columnData) in df.iterrows():
        if columnName == 'Yrkestitel':
            newRow['Combination'] = columnData.values.tolist()
        else:
            newRow.update({columnName: columnData.values.mean()})
    return newRow

def getAllCombinations(sourceDf, numberInList:int, numberToCombine:int):
    df = pd.DataFrame()
    my_list = list(range(0,numberInList))
    all_combinations = combinations(my_list, numberToCombine)

    for i in all_combinations:
        row = createNewRowFromIndex(sourceDf.iloc[list(i)])
        df = df._append(row, ignore_index = True)

    return df

if __name__ == '__main__':
    df = pd.read_excel('carl_test.xlsx')
    df_final = getAllCombinations(df, 3, 2)
    print(df_final)
    df_final.to_excel('carl_test_res.xlsx')