import pandas as pd
import matplotlib.pyplot as plt
import os 
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
#%%#-------------------------------------Funciones ---------------------------------------------
ruta = lambda x1: os.path.join(os.getcwd(),x1)
arrowFilter = lambda x1,x2: True if  not x1 in x2 else False
isClassStr = lambda x: True if str(type(x)) == "<class 'str'>" else False

def Str2Float(string):
    string = string.replace(",","")
    return float(string)
    

#%%#---------------------------------- Importar los datos --------------------------------------

fileData = ruta(os.path.join("data","farmData.csv"))
df = pd.read_csv(fileData,encoding= 'unicode_escape')

#%%#--------------------------------- Acomodar los datos --------------------------------------
dfCopy = df.copy()

for i in df.keys():
    if arrowFilter(i,["LGD Code","LGD Name"]) and isClassStr(df[i][0]):
        dfCopy[i] = [Str2Float(value) for pos,value in enumerate(df[i])]

X = dfCopy.drop(["LGD Code","LGD Name","Average"], axis = 1)
y = dfCopy["Average"]    
#%%#-------------------------------- regression lineal sin procesamiento ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

algth = linear_model.LinearRegression()
algth.fit(X_train,y_train)
print(algth.score(X_test,y_test))


