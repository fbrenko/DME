# Importamos las librerías necesarias
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Cargamos los datos de ejemplo
data = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# Definimos las variables independientes y la variable dependiente
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']

# Creamos el modelo de regresión logística
model = LogisticRegression()

# Entrenamos el modelo
model.fit(X, y)

# Hacemos una predicción con datos nuevos
new_data = [[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3], [7.3, 2.8, 6.3, 1.8]]
predicted = model.predict(new_data)

# Imprimimos los resultados
print(predicted)
