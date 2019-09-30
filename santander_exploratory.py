# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# Carga datos
data = pd.read_csv('data/train.csv')
data.set_index("ID_code", inplace=True)  # el id cómo índice

# Ver balanceo
target_count = data['target'].value_counts()
print('Clase 0:', target_count[0])
print('Clase 1:', target_count[1])
print('Proporción:', round(target_count[0] / target_count[1], 2), ': 1')
target_count.plot(kind='bar', title='Cantidad (target)')

# Separar el target
dataset = data.iloc[:, 1:]  # Todas las columnas menos la primera
target = data.iloc[:, :1]  # El target está en la primer columna

# Describe
describe = dataset.describe(include='all')

# Verificando valores nulos
dataset.isnull().values.any()

# Remover variables correlacionadas
corr_matrix = dataset.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
dataset.drop(dataset[to_drop], axis=1)  # No hay ninguna...
