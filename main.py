import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

# Cargar el conjunto de datos MNIST desde sklearn.datasets
mnist = fetch_openml('mnist_784', version=1)
x, y = mnist['data'], mnist['target']

df_x = pd.DataFrame(x)
df_y = pd.DataFrame(y)

# Obtener la mitad de las filas
mitad_filas = len(df_x) // 2

# Dividir el DataFrame a la mitad
df_x_columns_1 = df_x.iloc[:mitad_filas].copy()
df_x_columns_2 = df_x.iloc[mitad_filas:].copy()

df_y_1 = df_y.iloc[:mitad_filas]
df_y_2 = df_y.iloc[mitad_filas:]

# Convertir las filas en listas y asignarlas a una nueva columna
df_x_columns_1['x1'] = df_x_columns_1.apply(lambda row: row.tolist(), axis=1)
df_x_columns_2['x2'] = df_x_columns_2.apply(lambda row: row.tolist(), axis=1)


# Eliminar las columnas originales
df_x_columns_1.drop(df_x_columns_1.columns[:-1], axis=1, inplace=True)
df_x_columns_2.drop(df_x_columns_2.columns[:-1], axis=1, inplace=True)

# Resetear los índices del DataFrame df_x_columns_2
df_x_columns_2.reset_index(drop=True, inplace=True)
df_y_2.reset_index(drop=True, inplace=True)

# Concatenación del dataframe
result_concat_col = pd.concat([df_x_columns_1, df_x_columns_2, df_y_1, df_y_2], axis=1)

result_concat_col.columns.values[2:4] = ['y1', 'y2']
result_concat_col['son_iguales'] = (result_concat_col['y1'] == result_concat_col['y2'])

print(result_concat_col.head())
