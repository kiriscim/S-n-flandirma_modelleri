# Siniflandirma_modelleri
# Siniflandırma Modelleri

import numpy as np
import pandas as pd 
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

# Veri Seti Hikayesi ve Problem: Şeker Hastalığı Tahmini

df = pd.read_csv("./diabetes.csv")

df.head()

# Lojistik Regresyon (Logistic Regression)

# Model & Tahmin

df["Outcome"].value_counts()

df.describe().T

y = df["Outcome"]
X = df.drop(["Outcome"], axis = 1)

y.head()

X.head()

loj_model = LogisticRegression(solver = "liblinear").fit(X,y)

loj_model.intercept_

loj_model.coef_

loj_model.predict(X)[0:10]

y[0:10]

y_pred = loj_model.predict(X)

confusion_matrix(y, y_pred)

accuracy_score(y, y_pred)

print(classification_report(y, y_pred))

loj_model.predict_proba(X)[0:10]

logit_roc_auc = roc_auc_score(y, loj_model.predict(X))
fpr, tpr, thresholds = roc_curve(y, loj_model.predict_proba(X)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# Model Tuning (Model Doğrulama)

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.30, 
                                                    random_state=42)

loj_model = LogisticRegression(solver = "liblinear").fit(X_train,y_train)

y_pred = loj_model.predict(X_test)

print(accuracy_score(y_test, y_pred))

cross_val_score(loj_model, X_test, y_test, cv = 10).mean()

# K-En Yakın Komşu (KNN)

df.head()

y = df["Outcome"]
X = df.drop(['Outcome'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)

# Model & Tahmin

knn_model = KNeighborsClassifier().fit(X_train, y_train)

knn_model

y_pred = knn_model.predict(X_test)

accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))

# Model Tuning

knn = KNeighborsClassifier()

np.arange(1,50)

knn_params = {"n_neighbors": np.arange(1,50)}

knn_cv_model = GridSearchCV(knn, knn_params, cv = 10).fit(X_train, y_train)

knn_cv_model.best_score_

knn_cv_model.best_params_

#final model

knn_tuned = KNeighborsClassifier(n_neighbors = 11).fit(X_train, y_train)

y_pred = knn_tuned.predict(X_test)

accuracy_score(y_test, y_pred)

knn_tuned.score(X_test, y_test)

# Destek Vektör Makineleri (SVM)

df.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)

# Model & Tahmin

svm_model = SVC(kernel = "linear").fit(X_train, y_train)

svm_model

y_pred = svm_model.predict(X_test)

accuracy_score(y_test, y_pred)

# Model Tuning

svm = SVC()

svm_params = {"C": np.arange(1,10), "kernel": ["linear","rbf"]}

svm_cv_model = GridSearchCV(svm, svm_params, cv = 5, n_jobs = -1, verbose = 2).fit(X_train, y_train)

svm_cv_model.best_score_

svm_cv_model.best_params_

#final model

svm_tuned = SVC(C = 2, kernel = "linear").fit(X_train, y_train)

y_pred = svm_tuned.predict(X_test)

accuracy_score(y_test, y_pred)

# Yapay Sinir Ağları (Çok Katmanlı Algılayıcılar)

df.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)




scaler = StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)

scaler.fit(X_test)
X_test = scaler.transform(X_test)

# Model & Tahmin

mlpc_model = MLPClassifier().fit(X_train, y_train)

mlpc_model.coefs_

?mlpc_model

y_pred = mlpc_model.predict(X_test)

accuracy_score(y_test, y_pred)

# Model Tuning

mlpc_params = {"alpha":[1,5, 0.1,0.01, 0.03, 0.005, 0.0001],
              "hidden_layer_sizes": [(10,10), (100,100,100), (100,100), (3,5)]}

mlpc = MLPClassifier(solver = "lbfgs", activation = "logistic")

mlpc_cv_model = GridSearchCV(mlpc, mlpc_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train, y_train)

mlpc_cv_model

mlpc_cv_model.best_params_

#finalmodel

mlpc_tuned = MLPClassifier(solver = "lbfgs",activation='logistic', alpha = 1, hidden_layer_sizes = (3,5)).fit(X_train, y_train)

y_pred = mlpc_tuned.predict(X_test)

accuracy_score(y_test, y_pred)

# CART (Classification and Regression Tree)

df.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)

# Model & Tahmin

cart_model = DecisionTreeClassifier().fit(X_train, y_train)

cart_model

y_pred = cart_model.predict(X_test)

accuracy_score(y_test, y_pred)

# Model Tuning

cart = DecisionTreeClassifier()

cart_params = {"max_depth": [1,3,5,8,10],
              "min_samples_split": [2,3,5,10,20,50]}

cart_cv_model = GridSearchCV(cart, cart_params, cv = 10, n_jobs = -1, verbose =2).fit(X_train, y_train)

cart_cv_model.best_params_

#final model

cart_tuned = DecisionTreeClassifier(max_depth = 5, min_samples_split = 20).fit(X_train, y_train)

y_pred = cart_tuned.predict(X_test)

accuracy_score(y_test, y_pred)

# Random Forests

df.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)

# Model & Tahmin

rf_model = RandomForestClassifier().fit(X_train, y_train)

rf_model

y_pred = rf_model.predict(X_test)
accuracy_score(y_test, y_pred)

# Model Tuning

rf = RandomForestClassifier()

rf_params = {"n_estimators": [100,200,500,1000],
            "max_features": [3,5,7,8],
            "min_samples_split": [2,5,10,20]}

rf_cv_model = GridSearchCV(rf, rf_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train, y_train)

rf_cv_model.best_params_

#final model

rf_tuned = RandomForestClassifier(max_features = 8, 
                                  min_samples_split = 5, 
                                  n_estimators = 500).fit(X_train, y_train)

y_pred = rf_tuned.predict(X_test)

accuracy_score(y_test, y_pred)

#degisken onem duzeyleri

rf_tuned

feature_imp = pd.Series(rf_tuned.feature_importances_,
                        index=X_train.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Değişken Önem Skorları')
plt.ylabel('Değişkenler')
plt.title("Değişken Önem Düzeyleri")
plt.show()

# Gradient Boosting Machines

# Model & Tahmin

gbm_model = GradientBoostingClassifier().fit(X_train, y_train)

?gbm_model

y_pred = gbm_model.predict(X_test)

accuracy_score(y_test, y_pred)

# Model Tuning

gbm = GradientBoostingClassifier()

gbm_params = {"learning_rate": [0.1, 0.01, 0.001, 0.05],
             "n_estimators": [100, 300, 500, 1000],
             "max_depth":[2,3,5,8]}

gbm_cv_model= GridSearchCV(gbm, gbm_params, 
                           cv = 10, 
                           n_jobs = -1, verbose = 2).fit(X_train, y_train)

gbm_cv_model.best_params_

#final model

gbm_tuned = GradientBoostingClassifier(learning_rate = 0.01,
                                       max_depth = 5, 
                                       n_estimators = 500).fit(X_train, y_train)

y_pred = gbm_tuned.predict(X_test)
accuracy_score(y_test, y_pred)

feature_imp = pd.Series(gbm_tuned.feature_importances_,
                        index=X_train.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Değişken Önem Skorları')
plt.ylabel('Değişkenler')
plt.title("Değişken Önem Düzeyleri")
plt.show()

# XGBoost

# Model & Tahmin

!pip install xgboost

from xgboost import XGBClassifier

xgb_model = XGBClassifier().fit(X_train, y_train)

?xgb_model

y_pred = xgb_model.predict(X_test)

accuracy_score(y_test, y_pred)

# Model Tuning

xgb = XGBClassifier()

xgb_params = {"n_estimators": [100, 500, 1000],
             "subsample":[0.6,0.8,1],
             "max_depth":[3,5,7],
             "learning_rate":[0.1,0.001,0.01]}

xgb_cv_model = GridSearchCV(xgb, xgb_params, cv = 10, 
                            n_jobs = -1, verbose = 2).fit(X_train, y_train)

xgb_cv_model.best_params_

xgb_tuned = xgb = XGBClassifier(learning_rate= 0.001, 
                                max_depth= 7, 
                                n_estimators= 500, 
                                subsample= 0.6).fit(X_train, y_train)

y_pred = xgb_tuned.predict(X_test)

accuracy_score(y_test,y_pred)

feature_imp = pd.Series(xgb_tuned.feature_importances_,
                        index=X_train.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Değişken Önem Skorları')
plt.ylabel('Değişkenler')
plt.title("Değişken Önem Düzeyleri")
plt.show()

# Light GBM

# Model & Tahmin

!pip install lightgbm

!conda install -c conda-forge lightgbm

from lightgbm import LGBMClassifier

lgbm_model = LGBMClassifier().fit(X_train, y_train)

?lgbm_model

y_pred = lgbm_model.predict(X_test)

accuracy_score(y_test,y_pred)

# Model Tuning

lgbm = LGBMClassifier()

lgbm_params = {"learning_rate": [0.001, 0.01, 0.1],
              "n_estimators": [200, 500, 100],
              "max_depth":[1,2,35,8]}

lgbm_cv_model = GridSearchCV(lgbm,lgbm_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train, y_train)

lgbm_cv_model.best_params_

lgbm_tuned = LGBMClassifier(learning_rate= 0.01, 
                            max_depth= 1, 
                            n_estimators= 500).fit(X_train, y_train)

y_pred = lgbm_tuned.predict(X_test)

accuracy_score(y_test, y_pred)

feature_imp = pd.Series(lgbm_tuned.feature_importances_,
                        index=X_train.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Değişken Önem Skorları')
plt.ylabel('Değişkenler')
plt.title("Değişken Önem Düzeyleri")
plt.show()

# CatBoost

# Model & Tahmin

!pip install catboost

from catboost import CatBoostClassifier

catb_model = CatBoostClassifier().fit(X_train, y_train, verbose = False)

?catb_model

y_pred = catb_model.predict(X_test)

accuracy_score(y_test, y_pred)

# Model Tuning

catb = CatBoostClassifier()

catb_params = {"iterations":[200,500,100],
              "learning_rate":[0.01, 0.03, 0.1],
              "depth":[4,5,8]}

catb_cv_model = GridSearchCV(catb, catb_params, 
                             cv = 5, n_jobs = -1, verbose = 2).fit(X_train, y_train)

catb_cv_model.best_params_

catb_tuned = CatBoostClassifier(depth= 8, 
                                iterations= 200, 
                                learning_rate= 0.03).fit(X_train, y_train)

y_pred = catb_tuned.predict(X_test)

accuracy_score(y_test, y_pred)

feature_imp = pd.Series(catb_tuned.feature_importances_,
                        index=X_train.columns).sort_values(ascending=False)

sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Değişken Önem Skorları')
plt.ylabel('Değişkenler')
plt.title("Değişken Önem Düzeyleri")
plt.show()

# Tüm Modellerin Karşılaştırılması

modeller = [
    knn_tuned,
    loj_model,
    svm_tuned,
    mlpc_tuned,
    cart_tuned,
    rf_tuned,
    gbm_tuned,
    catb_tuned,
    lgbm_tuned,
    xgb_tuned]

sonuc = []
sonuclar = pd.DataFrame(columns= ["Modeller","Accuracy"])

for model in modeller:
    isimler = model.__class__.__name__
    y_pred = model.predict(X_test)
    dogruluk = accuracy_score(y_test, y_pred)
    sonuc = pd.DataFrame([[isimler, dogruluk*100]], columns= ["Modeller","Accuracy"])
    sonuclar = sonuclar.append(sonuc)

sns.barplot(x= 'Accuracy', y = 'Modeller', data=sonuclar, color="r")
plt.xlabel('Accuracy %')
plt.title('Modellerin Doğruluk Oranları');

sonuclar

# Daha Başka Ne Yapılabilir?

1. Değişken türetme / değişken mühendisliği
2. Değişken seçme
3. Otomatik ML
4. Model deployment

