import pandas as pd
from itertools import combinations

def createNewRowFromIndex(df:pd.DataFrame):
    """Takes pandas dataframe with multiple indexes and returns summarized row"""
    df = df.T
    newRow = {}
    for (columnName, columnData) in df.iterrows():
        if columnName == 'Yrkestitel':
            newRow['Combination'] = columnData.values.tolist()
        else:
            newRow.update({columnName: columnData.values.mean()})
    return newRow

def getAllCombinations(sourceDf:pd.DataFrame, numberInList:int, numberToCombine:int):
    df = pd.DataFrame()
    my_list = list(range(0, numberInList))
    all_combinations = combinations(my_list, numberToCombine)
    for i in all_combinations:
        row = createNewRowFromIndex(sourceDf.iloc[list(i)])
        df = df._append(row, ignore_index = True)
    return df

def prepareDataFrame(source_df:pd.DataFrame, colName = 'Max'):
    firstMax = source_df.T.iloc[1:,:].idxmax()
    for i, element in enumerate(firstMax):
        source_df.at[i, element] = 0
    secondMax = source_df.T.iloc[1:,:].idxmax()
    doubleMax = []
    for i, element in enumerate(firstMax):
        doubleMax.append([element, secondMax[i]])
    source_df[colName] = doubleMax    
    return source_df

def generateTrainingData(sourcePath:str, targetPath:str, fraction:float, sourceSheetNames = [0]):
    """Reads an excel, generates new excel with training and test data"""
    df_return = pd.DataFrame()
    for sourceSheetName in sourceSheetNames:
        df = pd.read_excel(sourcePath, sheet_name = sourceSheetName)
        df_double = prepareDataFrame(getAllCombinations(df, df.shape[0], 2), sourceSheetName)
        df_tripple = prepareDataFrame(getAllCombinations(df, df.shape[0], 3), sourceSheetName)
        df_concat = pd.concat([df_double, df_tripple], ignore_index = True)

        ##if not 'Combination' in df_return.columns:
        df_return['Combination'] = df_concat['Combination']
        df_return[sourceSheetName] = df_concat[sourceSheetName]

    ## df_train is 80% of total rows
    df_train = df_return.sample(frac = fraction)
    ## df_test is the rest (i.e 20% of total rows)
    df_test = df_return.drop(df_train.index)

    writer = pd.ExcelWriter(targetPath)
    df_train.to_excel(writer, columns = sourceSheetNames.insert(0, 'Combination'), sheet_name = 'train', index = False)
    df_test.to_excel(writer, columns = sourceSheetNames.insert(0, 'Combination'), sheet_name = 'test', index = False)
    writer.close()
    return 

if __name__ == '__main__':
    sheetNames = pd.ExcelFile('carl_test.xlsx').sheet_names
    generateTrainingData('carl_test.xlsx', 'training_data.xlsx', 0.8, sheetNames)

    """ df_test = pd.read_excel('carl_test.xlsx')
    test1 = getAllCombinations(df_test, df_test.shape[0], 3)
    test2 = getAllCombinations(df_test, df_test.shape[0], 3)
    print(test1)
    print(test2) """
