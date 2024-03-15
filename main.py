import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

# Cargar el conjunto de datos MNIST desde sklearn.datasets
mnist = fetch_openml('mnist_784', version=1)
x, y = mnist['data'], mnist['target']

# Convertir las etiquetas a enteros
y = y.astype(int)

# Crear DataFrame con los datos completos
mnist_df = pd.DataFrame(data=x, columns=[f"Pixel_{i}" for i in range(x.shape[1])])
mnist_df['Label'] = y  # Agregar columna de etiquetas

# Seleccionar aleatoriamente 200 datos
sample_df = mnist_df.sample(n=200, random_state=42)

# Seleccionar dos columnas para realizar el producto cartesiano
selected_columns = ['Pixel_100', 'Pixel_200']

# Calcular el producto cartesiano entre las dos columnas seleccionadas
cartesian_product = [(a, b, a*b) for a in sample_df[selected_columns[0]] for b in sample_df[selected_columns[1]]]

# Crear un nuevo DataFrame con los resultados del producto cartesiano
cartesian_df = pd.DataFrame(cartesian_product, columns=selected_columns + ['Product'])

# Verificar si el producto está en alguna de las dos columnas originales y crear la columna 'Match'
cartesian_df['Match'] = cartesian_df['Product'].isin(sample_df[selected_columns[0]]) | cartesian_df['Product'].isin(sample_df[selected_columns[1]])

print("DataFrame con el producto cartesiano y la columna 'Match':")
print(cartesian_df)