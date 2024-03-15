import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

# Cargar el conjunto de datos MNIST desde sklearn.datasets
mnist = fetch_openml('mnist_784', version=1)
x, y = mnist['data'], mnist['target']

df_x = pd.DataFrame(x)
df_y = pd.DataFrame(y)

nuevo_df = df_x.iloc[:5].copy()

nuevo_df['x1'] = nuevo_df.apply(lambda row: row.tolist(), axis=1)
nuevo_df.drop(nuevo_df.columns[:-1], axis=1, inplace=True)

new_df = pd.DataFrame(columns=['Columna 1', 'Columna 2'])

# Recorrer el DataFrame de productos cartesianos
for valores1 in nuevo_df.itertuples(index=False):
    for valores2 in nuevo_df.itertuples(index=False):        # Verificar si ya se ha comparado alguna de estas combinaciones
        if idx1 != idx2 and not any((new_df['Columna 1'] == idx1) & (new_df['Columna 2'] == idx2)) and not any((new_df['Columna 1'] == idx2) & (new_df['Columna 2'] == idx1)):
            # Agregar la combinación al nuevo DataFrame
            new_df = new_df.append({'Columna 1': idx1, 'Columna 2': idx2}, ignore_index=True)

# Mostrar el nuevo DataFrame
print(new_df)
