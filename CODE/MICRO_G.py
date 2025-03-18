#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:30:17 2025

@author: beto
"""

# For Data Manipulation
import numpy as np
import pandas as pd

# For Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For Model Selection & Hyperparameter Tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# For Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor, plot_importance

# For Feature Selection
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline

# For Model Evaluation & Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, ConfusionMatrixDisplay, classification_report, 
    roc_auc_score, roc_curve
)
from scipy.stats import chi2
from scipy.stats import norm








df = pd.read_excel("/Users/beto/Desktop/cartera.xlsx")

df.columns = df.columns.str.strip()
print(df.columns.tolist())

df['NIVEL POBREZA'] = df['NIVEL POBREZA'].fillna(method='ffill')

categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    df[col] = df[col].fillna("Desconocido")

df["RURAL"] = df["RURAL"].apply(lambda x: 1 if x == "RURAL" else 0)
df["PERÍODO GRACÍA"] = df["PERÍODO GRACÍA"].apply(lambda x: 1 if x == "SI" else 0)


# Objetivo Logit
df["IMPAGO"] = df["ATRASO > 90"].apply(lambda x: 1 if x > 0 else 0)

# Analisis de correlación
correlation_matrix = df.corr(numeric_only=True)
correlation = correlation_matrix["IMPAGO"].sort_values(ascending=False)

print(correlation)


# Efecto de variables categóricas
categorical_vars = ["ESTADO CIVIL", "SEXO", "TIPO PRÉSTAMO", "NIVEL POBREZA", 
                    "ACTIVIDAD ECONÓMICA", "TIPO GARANTÍA",
                    "CALIFICACION", "CALIF CONTABLE","TIPO AMORTIZACIÓN"]

impago_rates = {}
for var in categorical_vars:
    if var in df.columns:
        impago_rates[var] = df.groupby(var)["IMPAGO"].mean().sort_values(ascending=False)

print(impago_rates)


df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)





# Logit setup

features = ['IMPAGO',
       'TASA EFECTIVA', 
       'VALOR CUOTA', 
       'INTERÉS CUOTA ACTIVA',
       'CAPITAL CUOTA ACTIVA', 
       'INTERÉS CUOTA VENCIDA',
       'EDAD CLIENTE',
       'RURAL', 
#       'VALOR GARANTÍA', 
       'CICLO RENOVACIÓN',
       'ESTADO CIVIL_SOLTERO',
       'ESTADO CIVIL_UNIÓN LIBRE',
#       'ESTADO CIVIL_VIUDO', 
#       'SEXO_M',
#       'TIPO PRÉSTAMO_CREDI RURAL', 
       'TIPO PRÉSTAMO_CREDIAMIGO',
#       'TIPO PRÉSTAMO_DIRECTO', 
#       'TIPO PRÉSTAMO_PREFERENCIAL',
#       'TIPO PRÉSTAMO_RENOVACIÓN ANTICIPADA', 
#       'NIVEL POBREZA_B',
       'ACTIVIDAD ECONÓMICA_B', 
#       'ACTIVIDAD ECONÓMICA_C',
#       'ACTIVIDAD ECONÓMICA_D', 
#       'ACTIVIDAD ECONÓMICA_E',
       'ACTIVIDAD ECONÓMICA_F', 
#       'ACTIVIDAD ECONÓMICA_G',
#       'ACTIVIDAD ECONÓMICA_H', 
       'ACTIVIDAD ECONÓMICA_I',
#       'ACTIVIDAD ECONÓMICA_J', 
#       'ACTIVIDAD ECONÓMICA_K',
#       'ACTIVIDAD ECONÓMICA_L', 
       'CALIFICACION_A-2', 
#       'CALIFICACION_E',
#       'CALIFICACION_B-1', 
#       'CALIFICACION_C-1', 
#       'CALIFICACION_D',
       'CALIF CONTABLE_MICROCREDITO'
#       'CALIF CONTABLE_MICROCREDITO REESTRUCTURADO',
#       'CALIF CONTABLE_MICROCREDITO REFINANCIADO'
]



available_features = [col for col in features if col in df.columns]
df_model = df[available_features]
df_model = df_model.astype(int)



###############################################################################
######################### Logistic regression setup ###########################
###############################################################################

X = df_model.drop(columns=['IMPAGO'])
y = df_model['IMPAGO']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)




# Calculo de mejor regularizador en base cross-validation

#### LogReg base
log_clf = LogisticRegression(
    random_state=42, 
    max_iter=500, 
    class_weight="balanced", 
    penalty="l2"
)

#### Busqueda de Hiperparametros
param_grid = {
    "C": np.logspace(-4, 4, 20)  # Pruebar valores desde 10⁻⁴ hasta 10⁴
}

grid_search = GridSearchCV(
    log_clf, 
    param_grid, 
    cv=5,  
    scoring="roc_auc",  
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_lambda = 1 / grid_search.best_params_['C']
#print(f"Mejor λ encontrado: {best_lambda}")




log_clf = LogisticRegression(
    random_state=42, 
    max_iter=500, 
    class_weight="balanced", 
    penalty="l2",       # Ridge
    C=1/best_lambda        # Regularization strength (lower = stronger penalty)
).fit(X_train, y_train)



coefs = log_clf.coef_[0] 
variables = list(X_train.columns)

# Estimate standard errors using an approximation
preds = log_clf.predict_proba(X_train)[:, 1]  
W_diag = preds * (1 - preds)  

try:
    XTWX = X_train.T @ (X_train * W_diag[:, np.newaxis])  # Proper alignment
    cov_matrix = np.linalg.inv(XTWX)  # Covariance matrix
    std_errors = np.sqrt(np.diag(cov_matrix))  # Standard errors

    # Compute 95% confidence intervals
    z_score = norm.ppf(0.975)  # 1.96 for 95% confidence
    conf_lower = coefs - z_score * std_errors
    conf_upper = coefs + z_score * std_errors

    z_scores = coefs / std_errors
    p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))  # Two-tailed test

    # Summary DataFrame
    summary_df = pd.DataFrame({
        "Variable": variables,
        "Coefficient": coefs,
        "P-Value": p_values,    
        "Std. Error": std_errors,
        "Conf. Interval Lower": conf_lower,
        "Conf. Interval Upper": conf_upper
    })

except np.linalg.LinAlgError:
    summary_df = pd.DataFrame({
        "Variable": variables,
        "Coefficient": coefs,
        "P-Value": [np.nan] * len(coefs),    
        "Std. Error": [np.nan] * len(coefs),
        "Conf. Interval Lower": [np.nan] * len(coefs),
        "Conf. Interval Upper": [np.nan] * len(coefs)
    })


###############################################################################
######################### (IM)BALANCE PROTOCOLS #############################
###############################################################################

# In case of persistent multicolinearity and or singularity in the Hessian, essai this protocols 
# to check for improvement details


for col in X_train.columns:
    crosstab = pd.crosstab(X_train[col], y_train)
    if (crosstab.nunique(axis=1) == 1).all():
        print(f"⚠️ Warning: {col} perfectly predicts the target (IMPAGO). Consider removing it.")
        
low_variance_cols = X_train.var()[X_train.var() < 1e-5].index
print(f"🔍 Low variance features to remove: {list(low_variance_cols)}")




y_pred = log_clf.predict(X_test)

log_cm = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=log_clf.classes_)

log_disp.plot(values_format='')
plt.show()

    
# Check for multicolinearity
corr_matrix = X.corr().abs()

high_corr = np.where(corr_matrix > 0.90)
high_corr_pairs = [(corr_matrix.index[i], corr_matrix.columns[j]) for i, j in zip(*high_corr) if i != j]

print("Highly correlated feature pairs:", high_corr_pairs)

# Duplicates
duplicate_columns = X.T.duplicated()
X = X.loc[:, ~duplicate_columns] 
print("Duplicated varis:", duplicate_columns)




#Class balance
df_model['IMPAGO'].value_counts(normalize=True)
# Imbalance is not severe (90%-10% would be highly severe)



# Create classification report for logistic regression model
target_names = ['Deuda no recuperable', 'Deuda recuperable']
print(classification_report(y_test, y_pred, target_names=target_names))






###############################################################################
######################### MACHINE LEARNING MODELS #############################
###############################################################################

#Decision tree

tree = DecisionTreeClassifier(random_state=0)
cv_params = {'max_depth':[4, 6, 8, None], 
             'min_samples_leaf': [2, 5, 1], 
             'min_samples_split': [2, 4, 6]
}

scoring = {
    'roc_auc': 'roc_auc',
    'precision': 'precision',
    'f1': 'f1',
    'recall': 'recall',
    'accuracy': 'accuracy'
}


tree1 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')


import time
import timeit

def tree_function():
    time.sleep(2)
    tree1.fit(X_train, y_train) #Tree fiting 

execution_time = timeit.timeit(lambda: tree_function(), number=1) 
print(f"Execution time: {execution_time:.2f} seconds")





def make_results(model_name: str, model_object, metric: str):
    metric_dict = {
        'auc': 'mean_test_roc_auc',
        'precision': 'mean_test_precision',
        'recall': 'mean_test_recall',
        'f1': 'mean_test_f1',
        'accuracy': 'mean_test_accuracy'
    }

    if metric not in metric_dict:
        raise ValueError(f"Invalid metric '{metric}'. Choose from {list(metric_dict.keys())}")

    cv_results = pd.DataFrame(model_object.cv_results_)

    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract the best scores
    auc = best_estimator_results.get('mean_test_roc_auc', None)
    f1 = best_estimator_results.get('mean_test_f1', None)
    recall = best_estimator_results.get('mean_test_recall', None)
    precision = best_estimator_results.get('mean_test_precision', None)
    accuracy = best_estimator_results.get('mean_test_accuracy', None)

    table = pd.DataFrame({
        'model': [model_name],
        'auc': [auc],
        'f1': [f1],
        'recall': [recall],
        'precision': [precision],
        'accuracy': [accuracy]
    })

    return table

tree1_cv_results = make_results('decision tree cv', tree1, 'auc')



# Random forest

rf = RandomForestClassifier(random_state=0)

cv_params = {'max_depth': [3,5, None], 
             'max_features': [1.0],
             'max_samples': [0.7, 1.0],
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'n_estimators': [300, 500],
}

rf1 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')
rf1.fit(X_train, y_train) 


rf1_cv_results = make_results('random forest cv', rf1, 'auc')
print(tree1_cv_results)
print(rf1_cv_results)


# Importance features

tree1_importances = pd.DataFrame(tree1.best_estimator_.feature_importances_,
                                 columns=['gini_importance'],
                                 index=X.columns
                                )
tree1_importances = tree1_importances.sort_values(by='gini_importance',ascending=False)
# Only extract the features with importances > 0
tree2_importances = tree1_importances[tree1_importances['gini_importance'] != 0]
tree2_importances





plt.figure(figsize=(18, 15))
sns.barplot(data=tree1_importances, x="gini_importance", y=tree1_importances.index, orient='h')
plt.title("Árbol de decisión: Importancia de Rasgos para la Probabilidad de Incumplimiento",fontsize=18)
plt.ylabel("Rasgo",fontsize=18)
plt.xlabel("Grado de importancia",fontsize=18)
plt.show()









plt.figure(figsize=(18, 15))
feat_impt = rf1.best_estimator_.feature_importances_ 
ind = np.argpartition(rf1.best_estimator_.feature_importances_, -10)[-10:]
feat = X.columns[ind]
# Filter `feat_impt` to consist of top 10 feature importances
feat_impt = feat_impt[ind]
y_df = pd.DataFrame({"Feature":feat,"Importance":feat_impt})
y_sort_df = y_df.sort_values("Importance")
fig = plt.figure()
ax1 = fig.add_subplot(111)
y_sort_df.plot(kind='barh',ax=ax1,x="Feature",y="Importance")




ax1.set_title("Random Forest: Rasgos clave para Impago",fontsize=12)
ax1.set_ylabel("Rasgos")
ax1.set_xlabel("Importancia")
plt.show()






















