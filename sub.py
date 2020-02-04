import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




dataset = pd.read_csv("train.csv")
df = pd.read_csv("test.csv")
pid = df['PassengerId'].values
pid = pd.DataFrame(pid,columns = ["PassengerId"])
dataset.Age.mean()

dataset.Age.fillna(dataset.Age.median(), inplace=True)
df.Age.fillna(df.Age.median(), inplace=True)

dataset.Embarked.mode()
dataset.Embarked.fillna('S', inplace = True)




dataset.info()
df.info()
df.Fare.count()
df.Fare.fillna(df.Fare.median(), inplace = True)

sns.heatmap(df.isnull(), cmap = "inferno_r")
dataset[['Sex','Survived']].groupby('Sex').mean().sort_values(by = "Survived", ascending = "True")

smap = {'male' : 0 , 'female' : 1}
dataset['Sex'] = dataset['Sex'].map(smap).astype(int)
df['Sex'] = df['Sex'].map(smap)


dataset[['Embarked','Survived']].groupby('Embarked').mean().sort_values(by = 'Survived' , ascending = 'True')

emap = {'S' : 0 , 'Q' : 1 , 'C' : 2}

dataset['Embarked'] = dataset['Embarked'].map(emap).astype(int)
df['Embarked'] = df['Embarked'].map(emap).astype(int)
dataset.describe()

sns.heatmap(dataset.isnull(),cmap = 'inferno_r')
dataset.info()



df.info()
dataset.info()

df = df.drop(['Name','PassengerId','Ticket','Cabin'],axis = 1)
dataset = dataset.drop(['Name','PassengerId','Ticket','Cabin'],axis = 1)

dataset['Age'].describe()
sns.boxplot(x = dataset['Age'])

q1 = dataset['Age'].quantile(0.25)
q3 = dataset['Age'].quantile(0.75)
irq = q3 - q1
l = q1 - 1.5 * irq
h = q3 + 1.5 * irq
dataset = dataset.loc[(dataset['Age'] > l) & (dataset['Age'] < h)]

q1 = dataset['Fare'].quantile(0.25)
q3 = dataset['Fare'].quantile(0.75)
irq = q3 - q1
lw = q1 - 1.5 * irq
hi = q3 + 1.5 * irq
dataset = dataset.loc[(dataset['Fare'] > lw) & (dataset['Fare'] < hi)]

dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
df['Fare'] = df['Fare'].map(lambda i: np.log(i) if i > 0 else 0)




X_train = dataset.iloc[:,1:8].values
y_train = dataset.iloc[:,0].values
X_test = df.iloc[:, 0:7].values


for i in range(0,718):
    X_train[i][2] = X_train[i][2]//10


for i in range(0,418):
    X_test[i][2] = X_test[i][2]//10


'''
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categorical_features = [6])
X_train = one.fit_transform(X_train).toarray()
one = OneHotEncoder(categorical_features = [6])
X_test = one.fit_transform(X_test).toarray()







from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''



from sklearn.ensemble import RandomForestClassifier
cfr = RandomForestClassifier(n_estimators=100,criterion = 'gini',bootstrap = 'False',min_samples_leaf = 3,min_samples_split = 2)
cfr.fit(X_train,y_train)
rfacc = round(cfr.score(X_train,y_train)*100,2)
y_pred1 = cfr.predict(X_test)




from sklearn.tree import DecisionTreeClassifier
cf = DecisionTreeClassifier(criterion = "entropy",min_samples_leaf = 3,min_samples_split = 2,splitter = 'random')
cf.fit(X_train,y_train)
acc = round(cf.score(X_train,y_train)*100,2)
y_pred = cf.predict(X_test)




from sklearn.model_selection import GridSearchCV
accuracy = [{'criterion' : ['gini','entropy'],'splitter':['best','random'],"min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10]}
            ]

gsRFC = GridSearchCV(cf,param_grid = accuracy, cv=10, scoring="accuracy", n_jobs= -1)
gsRFC.fit(X_train,y_train)
gsRFC.best_params_
gsRFC.best_score_
dt_best = gsRFC.best_estimator_

accuracyRF = [{"max_depth": [None],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]
              }
        ]

rfgs =  GridSearchCV(cfr,param_grid = accuracyRF, cv=10, scoring="accuracy", n_jobs= -1)
rfgs.fit(X_train,y_train)
rfgs.best_params_
rfgs.best_score_
rf_best = rfgs.best_estimator_


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
n_neighbors = [6,7,8,9,10,11,12,14,16,18,20,22]
algorithm = ['auto']
weights = ['uniform', 'distance']
leaf_size = list(range(1,50,5))
hyperparams = {'algorithm': algorithm,
               'weights': weights,
               'leaf_size': leaf_size,
               'n_neighbors': n_neighbors}

gsKNN=GridSearchCV(estimator = knn, param_grid = hyperparams, verbose=1, cv=10, scoring = "roc_auc", n_jobs=-1)

gsKNN.fit(X_train, y_train)

knn_best = gsKNN.best_estimator_

gsKNN.best_params_
gsKNN.best_score_


knn = KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski', 
                           metric_params=None, n_jobs=1, n_neighbors=6, p=2, 
                           weights='uniform')
knn.fit(X_train, y_train)
y_predk = knn.predict(X_test)
 


from sklearn.model_selection import cross_val_score
cvs = cross_val_score(estimator = cf,X = X_train, y = y_train,cv = 10,n_jobs = -1)
cvs.mean()


cvrf =  cross_val_score(estimator = cfr,X = X_train, y = y_train,cv = 10,n_jobs = -1)
cvrf.mean()


cvk = cross_val_score(estimator = knn,X = X_train, y = y_train,cv = 10,n_jobs = -1)
cvk.mean()

from sklearn.ensemble import VotingClassifier
vc = VotingClassifier(estimators = [('rfc', rf_best),('dtc',dt_best)], voting = 'soft',n_jobs = -1)
vc.fit(X_train,y_train)
y_predvc = vc.predict(X_test)







y_pred = pd.DataFrame(y_pred,columns = ["Survived"])
y_pred1 = pd.DataFrame(y_pred1,columns = ["Survived"])
y_predk = pd.DataFrame(y_predk,columns = ["Survived"])
y_predvc = pd.DataFrame(y_predvc,columns = ["Survived"])

ans = pd.concat([pid,y_pred],axis = 1)
ans.to_csv("submission.csv",index = False)

ans1 = pd.concat([pid,y_pred1],axis = 1)
ans1.to_csv("sub.csv",index = False)

ansk = pd.concat([pid,y_predk],axis = 1)
ansk.to_csv("subk.csv",index = False)

ansvc = pd.concat([pid,y_predvc],axis = 1)
ansvc.to_csv("subvc.csv",index = False)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train,y_pred)



