# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from imblearn.under_sampling import RandomUnderSampler
import gc

SEED = 15
K_FOLD = 5
JOBS = 6

print('Leyendo dataset')
# Carga datos
data = pd.read_csv('data/train.csv')
data.set_index("ID_code", inplace=True)  # el id cómo índice
# Separar el target
dataset = data.iloc[:, 1:]  # Todas las columnas menos la primera
target = data.iloc[:, :1]  # El target está en la primer columna

print('\nSeparando en traint y test')
X_train, X_test, y_train, y_test = train_test_split(dataset, target, test_size=0.2, random_state=SEED, stratify=target)

print('\nBalanceo de clases\n')
# Ver balanceo
target_count = y_train['target'].value_counts()
print('Clase 0:', target_count[0])
print('Clase 1:', target_count[1])
print('Proporción:', round(target_count[0] / target_count[1], 2), ': 1\n')

# Balanceando clases de entrenamiento, metodo random unsersampling
target = y_train['target'].ravel()  # array con el target
balance = RandomUnderSampler(sampling_strategy='auto', random_state=SEED)
X_train_balance, y_train_balance = balance.fit_sample(X_train, y_train)
idx = balance.sample_indices_

X_train = X_train.iloc[idx]
y_train = y_train.iloc[idx]

# Ver balanceo nuevamente
target_count_balance = y_train['target'].value_counts()
print('Clase 0:', target_count_balance[0])
print('Clase 1:', target_count_balance[1])
print('Proporción:', round(target_count_balance[0] / target_count_balance[1], 2), ': 1')

# Se remueve lo que ya no se usa
del [data, dataset, target, X_train_balance, y_train_balance, target_count, target_count_balance]
gc.collect()

y_train = y_train['target'].ravel()  # array con el target

print('\nPreparando TPOT')

tpot = TPOTClassifier(max_time_mins=1080,
                      max_eval_time_mins=20,
                      scoring='roc_auc',
                      cv=K_FOLD,
                      periodic_checkpoint_folder='results/santander_checkpoints',
                      early_stop=10,
                      random_state=SEED,
                      n_jobs=JOBS,
                      verbosity=2)

tpot.fit(X_train, y_train)
print('\nroc_auc: {}'.format(tpot.score(X_test, y_test['target'].ravel())))

print('\nExportando resultados')
my_dict = list(tpot.evaluated_individuals_.items())
model_scores = pd.DataFrame()
for model in my_dict:
    model_name = model[0]
    model_info = model[1]
    cv_score = model[1].get('internal_cv_score')  # Pull out cv_score as a column (i.e., sortable)
    model_scores = model_scores.append({'model': model_name,
                                        'cv_score': cv_score,
                                        'model_info': model_info, },
                                       ignore_index=True)

model_scores = model_scores.sort_values('cv_score', ascending=False)

model_scores.to_csv('results/santander_pipelines_scores.csv', header=True, index=True)
tpot.export('results/santander_pipeline.py')

print('\nListo!')
