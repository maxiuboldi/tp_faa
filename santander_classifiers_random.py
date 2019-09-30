import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

SEED = 15
K_FOLD = 5
JOBS = 6
N_ITER = 1000
np.random.seed(SEED)

print('Leyendo dataset')
# Carga datos
data = pd.read_csv('data/train.csv')
data.set_index("ID_code", inplace=True)  # el id cómo índice
# Separar el target
dataset = data.iloc[:, 1:]  # Todas las columnas menos la primera
target = data.iloc[:, :1]  # El target está en la primer columna

print('\nSeparando en traint y test')
X_train, X_test, y_train, y_test = train_test_split(dataset, target, test_size=0.2, random_state=SEED, stratify=target)

print('\nExportando conjunto de datos')
X_train.to_csv('data/X_train_random.csv', header=True, index=True)
y_train.to_csv('data/y_train_random.csv', header=True, index=True)
X_test.to_csv('data/X_test_random.csv', header=True, index=True)
y_test.to_csv('data/y_test_random.csv', header=True, index=True)

print('\nPreparando pipeline')

y_train = y_train['target'].ravel()  # array con el target

pipe = Pipeline([('scl', SelectKBest(score_func=f_classif)),
                 ('clf', SVC(random_state=SEED))])

# esta forma de definir el espacio de búsqueda con una lista en RandomizedSearchCV está disponible a partir de la versión Version 0.22.dev0.
# pip install --pre -f https://sklearn-nightly.scdn8.secure.raxcdn.com scikit-learn
search_space = [{'clf': [LogisticRegression(random_state=SEED)],
                 'scl__k': [50, 100, 150, 200],
                 'clf__penalty': ['l1', 'l2'],
                 'clf__C': np.logspace(-4, 4, 50),
                 'clf__solver': ['liblinear'],
                 'clf__max_iter': range(100, 500, 50),
                 'clf__multi_class': ['auto']},
                {'clf': [RandomForestClassifier(random_state=SEED)],
                 'scl__k': [50, 100, 150, 200],
                 'clf__criterion': ['gini', 'entropy'],
                 'clf__bootstrap': [True, False],
                 'clf__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 'clf__max_depth': range(3, 30, 2),
                 'clf__min_samples_split': [2, 5, 10],
                 'clf__n_estimators': [4, 8, 16, 32, 64, 128, 256, 512]},
                {'clf': [AdaBoostClassifier(random_state=SEED)],
                 'scl__k': [50, 100, 150, 200],
                 'clf__n_estimators': [4, 8, 16, 32, 64, 128, 256, 512],
                 'clf__learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3]},
                {'clf': [GradientBoostingClassifier(random_state=SEED)],
                 'scl__k': [50, 100, 150, 200],
                 'clf__max_depth': range(3, 30, 2),
                 'clf__learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
                 'clf__subsample': [i / 10.0 for i in range(3, 10)],
                 'clf__n_estimators': [4, 8, 16, 32, 64, 128, 256, 512]},
                {'clf': [ExtraTreesClassifier(random_state=SEED)],
                 'scl__k': [50, 100, 150, 200],
                 'clf__bootstrap': [True, False],
                 'clf__max_depth': range(3, 30, 2),
                 'clf__min_samples_leaf': [1, 2, 4],
                 'clf__min_samples_split': [2, 5, 10],
                 'clf__n_estimators': [4, 8, 16, 32, 64, 128, 256, 512]},
                {'clf': [XGBClassifier(random_state=SEED)],
                 'scl__k': [50, 100, 150, 200],
                 'clf__max_depth': range(3, 30, 2),
                 'clf__learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
                 'clf__subsample': [i / 10.0 for i in range(3, 10)],
                 'clf__colsample_bytree': [i / 10.0 for i in range(2, 10)],
                 'clf__min_child_weight': range(1, 20, 2),
                 'clf__gamma': [i / 10.0 for i in range(0, 10)],
                 'clf__n_estimators': [4, 8, 16, 32, 64, 128, 256, 512]},
                {'clf': [LGBMClassifier(random_state=SEED)],
                 'scl__k': [50, 100, 150, 200],
                 'clf__num_leaves': list(range(8, 92, 4)),
                 'clf__min_data_in_leaf': [10, 20, 40, 60, 100],
                 'clf__max_depth': range(3, 30, 2),
                 'clf__learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
                 'clf__bagging_freq': [3, 4, 5, 6, 7],
                 'clf__bagging_fraction': np.linspace(0.6, 0.95, 10),
                 'clf__reg_alpha': np.linspace(0.1, 0.95, 10),
                 'clf__reg_lambda': np.linspace(0.1, 0.95, 10),
                 'clf__n_estimators': [4, 8, 16, 32, 64, 128, 256, 512]},
                {'clf': [SVC(random_state=SEED)],
                 'scl__k': [50, 100, 150, 200],
                 'clf__kernel': ['linear', 'rbf', 'poly'],
                 'clf__degree': [0, 1, 2, 3, 4, 5, 6],
                 'clf__gamma': ['scale', 'auto'],
                 'clf__decision_function_shape': ['ovo', 'ovr'],
                 'clf__C': np.linspace(0.001, 200, 50)}]

gs = RandomizedSearchCV(estimator=pipe,
                        param_distributions=search_space,
                        scoring='accuracy',
                        cv=K_FOLD,
                        n_jobs=JOBS,
                        verbose=1,
                        return_train_score=True,
                        n_iter=N_ITER)

print('\nAjustando Pipeline en la búsqueda con CV\n')
gs.fit(X_train, y_train)

# resultados del CV
cv_result = pd.DataFrame(gs.cv_results_)

# features finales y pesos
names = X_train.columns.values[gs.best_estimator_.named_steps["scl"].get_support()]
scores = gs.best_estimator_.named_steps["scl"].scores_[gs.best_estimator_.named_steps["scl"].get_support()]
names_scores = list(zip(names, scores))
features_sel = pd.DataFrame(names_scores, columns=['Feature', 'Scores'])

# predicción del resultado
y_pred = gs.predict(X_test)

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
with pd.ExcelWriter(r'results\resultados_santander_random.xlsx') as writer:
    conf_mat.to_excel(writer, sheet_name='Matriz_Confusion')
    class_rep.to_excel(writer, sheet_name='Reporte_Clasificacion')

joblib.dump(gs.best_estimator_, 'results/santander_model_random.pkl')

print('\nListo!')
