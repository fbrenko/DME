import pandas as pd # importo pandas para el tratamiento de data frames
import numpy as np  # import numpy para el manejo de arrays
from sklearn import linear_model # SKLEARN es la libreria de Machine Learning que implementa el algoritmo de Regresión Lineal

df = pd.read_csv('C:/Users/Fer/Documents/Fernando/archivospython/EjerciciosRegresion2.csv', sep=';')
df

#Elimino la varible dependiente
df1=df.drop('SueldoAnual',axis='columns')
df1=df1.drop('Sexo',axis='columns')

#Transformo variable Sexo categorica a Binaria
#dummies = pd.get_dummies(df1['Sexo'], prefix='Sexo')
#dummies

#Borro la variable Sexo Categorica
#df1=df.drop('Sexo',axis='columns')
#df1 = pd.concat([df1, dummies], axis=1)
df1

#Puedo pasar a Array y trabajar con DataFrame
#X = np.array(df1.to_numpy())
#y = np.array(df.SueldoAnual.to_numpy())
#reg.fit(X,y)

reg = linear_model.LinearRegression()
reg.fit(df1,df.SueldoAnual)


reg.coef_

reg.intercept_

reg.predict([[4, 5]])