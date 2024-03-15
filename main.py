import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

# Cargar el conjunto de datos MNIST desde sklearn.datasets
mnist = fetch_openml('mnist_784', version=1)
x, y = mnist['data'], mnist['target']

df_x = pd.DataFrame(x)
df_y = pd.DataFrame(y)
#print(df)


# Definir una función para convertir una fila en una lista de columnas
def row_to_columns(row):
    return [row[i:i+28] for i in range(0, len(row), 28)]

# Aplicar la función a cada fila y crear una nueva columna con listas de columnas
df_x_columns = df_x.apply(row_to_columns, axis=1)

# Obtener la mitad de las filas
mitad_filas = len(df_x_columns) // 2

# Dividir el DataFrame a la mitad
df_x_columns_1 = df_x_columns.iloc[:mitad_filas]
df_x_columns_2 = df_x_columns.iloc[mitad_filas:]

df_y_1 = df_y.iloc[:mitad_filas]
df_y_2 = df_y.iloc[mitad_filas:]

df_y_1.columns = ['target_1']
df_y_2.columns = ['target_2']

# Junto los cuatro en un solo data frame
merged_df = pd.concat([df_x_columns_1.reset_index(drop=True),
                       df_x_columns_2.reset_index(drop=True),
                       df_y_1.reset_index(drop=True),
                       df_y_2.reset_index(drop=True)], axis=1)

# Agregar una nueva columna con 1 si los valores son iguales, 0 si son distintos
merged_df['igualdad'] = merged_df.apply(lambda row: 1 if row['target_1'] == row['target_2'] else 0, axis=1)

# Mostrar el DataFrame con la nueva columna
print(merged_df.head())
