import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
import gc

SEED = 15
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

print('\nExportando conjunto de datos')
X_train.to_csv('data/X_train.csv', header=True, index=True)
y_train.to_csv('data/y_train.csv', header=True, index=True)
X_test.to_csv('data/X_test.csv', header=True, index=True)
y_test.to_csv('data/y_test.csv', header=True, index=True)

print('\nPreparando pipeline')

y_train = y_train['target'].ravel()  # array con el target

# Average CV score on the training set was:0.8865192171711967
exported_pipeline = make_pipeline(
    StandardScaler(),
    GaussianNB()
)

exported_pipeline.fit(X_train, y_train)
y_pred = exported_pipeline.predict(X_test)

# métricas para exportar
conf_mat = pd.DataFrame(confusion_matrix(y_test, y_pred), columns=['Clase_0', 'Clase_1'], index=['Clase_0', 'Clase_1'])
class_rep = pd.DataFrame.from_dict(classification_report(y_test, y_pred, target_names=['Clase_0', 'Clase_1'], output_dict=True))

print('\nMatriz de Confusión')
print(conf_mat)
print('\nReporte de Clasificación')
print(classification_report(y_test, y_pred, target_names=['Clase_0', 'Clase_1']))
print('\nAccuracy_score: {}'.format(accuracy_score(y_test, y_pred)))
print('\nROC_auc_score: {}'.format(roc_auc_score(y_test, y_pred)))

print('\nExportando resultados')
with pd.ExcelWriter(r'results\resultados_santander.xlsx') as writer:
    conf_mat.to_excel(writer, sheet_name='Matriz_Confusion')
    class_rep.to_excel(writer, sheet_name='Reporte_Clasificacion')

joblib.dump(exported_pipeline, 'results/santander_model.pkl')

print('\nListo!')
