import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('wine_quality.csv')
print(df.columns)
y = df['quality']
features = df.drop(columns = ['quality'])


## 1. Data transformation
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform(features)
## 2. Train-test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=99,test_size=0.2)
## 3. Fit a logistic regression classifier without regularization
from sklearn.linear_model import LogisticRegression
clf_no_reg=LogisticRegression(penalty= 'none' )
clf_no_reg.fit(x_train,y_train)
## 4. Plot the coefficients
predictors = features.columns
coefficients = clf_no_reg.coef_.ravel()
coef = pd.Series(coefficients,predictors).sort_values()
coef.plot(kind='bar', title = 'Coefficients (no regularization)')
plt.tight_layout()
plt.show()
plt.clf()
## 5. Training and test performance
from sklearn.metrics import f1_score
y_pred=clf_no_reg.predict(x_test)
y_predtr=clf_no_reg.predict(x_train)
print("Training score for model without regularisation",f1_score(y_train,y_predtr))
print("Testing score for model without regularisation",f1_score(y_test,y_pred))
## 6. Default Implementation (L2-regularized!)
from sklearn.linear_model import LogisticRegressionCV
clf_default=LogisticRegression()
# clf_default=LogisticRegressionCV(scoring='f1',cv=5,Cs=np.logspace(-3,2,100),max_iter=1000)
clf_default.fit(x_train,y_train)
y_pred=clf_default.predict(x_test)
y_predtr=clf_default.predict(x_train)
## 7. Ridge Scores
print("Training score for model with ridge regularisation",f1_score(y_train,y_predtr))
print("Testing score for model with ridge regularisation",f1_score(y_test,y_pred))
## 8. Coarse-grained hyperparameter tuning
training_array = []
test_array = []
C_array = [0.0001, 0.001, 0.01, 0.1, 1]
for x in C_array:
    clf = LogisticRegression(C = x)
    clf.fit(x_train, y_train)
    y_pred_test = clf.predict(x_test)
    y_pred_train = clf.predict(x_train)
    training_array.append(f1_score(y_train, y_pred_train))
    test_array.append(f1_score(y_test, y_pred_test))
## 9. Plot training and test scores as a function of C
plt.plot(C_array,training_array)
plt.plot(C_array,test_array)
plt.xscale('log')
plt.show()
plt.clf()

## 10. Making a parameter grid for GridSearchCV
tuning_C={'C':np.logspace(-4,-2,100)}

## 11. Implementing GridSearchCV with l2 penalty
from sklearn.model_selection import GridSearchCV
gs=GridSearchCV(clf,cv=5,param_grid=tuning_C,scoring='f1')
gs.fit(x_train,y_train)
## 12. Optimal C value and the score corresponding to it
optimal_C=gs.best_params_
print("optimal C",gs.best_params_)
print("Best score in regularization",gs.best_score_)
## 13. Validating the "best classifier"
clf_best_ridge=LogisticRegression(C=optimal_C['C'])
clf_best_ridge.fit(x_train,y_train)
y_pred=clf_best_ridge.predict(x_test)
y_predtr=clf_best_ridge.predict(x_train)
print("Best classifier training score:",f1_score(y_train,y_predtr))
print("Best classifier testing score:",f1_score(y_test,y_pred))
## 14. Implement L1 hyperparameter tuning with LogisticRegressionCV
from sklearn.linear_model import LogisticRegressionCV
clf_l1 = LogisticRegressionCV(Cs=np.logspace(-2, 2, 100), cv=5, penalty='l1', solver='liblinear', scoring='f1')

## 15. Optimal C value and corresponding coefficients

clf_l1.fit(X,y)
print("Best C value",clf_l1.C_,'\nThe Best Coefficients are:',clf_l1.coef_)
## 16. Plotting the tuned L1 coefficients
coefficients = clf_l1.coef_.ravel()
coef = pd.Series(coefficients,predictors).sort_values()

plt.figure(figsize = (12,8))
coef.plot(kind='bar', title = 'Coefficients for tuned L1')
plt.tight_layout()
plt.show()
plt.clf()
