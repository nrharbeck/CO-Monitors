
# import packages
import pandas as pd
from datetime import datetime
import numpy as np
import re
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

#The below file was pared down into important features from the full 2017 American Housing Survey dataset. The code for this can be found in R syntax in the Github repo.
COdf = pd.read_csv("https://nickharbeck.com/wp-content/uploads/2019/11/AHSCO.csv")
#The next few lines of code set all strings as numbers.
for col in COdf[['MONOXIDE','CONTROL','HEATTYPE','HEATFUEL','HOTWATER','ACPRIMARY','ACSECNDRY','HOA','RENTCNTRL','HHMAR','HHGRAD','HHRACE','INTLANG']]:
  COdf[col] = COdf[col].apply(lambda x: re.findall('[^0-9-]+([0-9-]+)', str(x)))
  COdf[col] = COdf[col].str.get(0)
  COdf[col] = pd.to_numeric(COdf[col])
  
#Here's a clean and smaller copy of the labels hosted on my site
AHS_LABELSNH = pd.read_csv("https://nickharbeck.com/wp-content/uploads/2019/11/ValueLabels.csv")
AHS_LABELSNH['Value'] = AHS_LABELSNH['Value'].str.extract('(\d+)', expand=False)

#The cleaned dataset is a copy of the original dataset that includes complete rows
df = COdf.fillna(-10)
df = df[(df['MONOXIDE']>= 0 ) & (df['HOA']>=0) & (df['YRBUILT']>=0) & (df['RENT']>=0) & (df['RENTCNTRL']>=0) & (df['HHMAR']>=0) & (df['HHGRAD']>=0) & (df['HHRACE']>=0) & (df['INTLANG']>=0) & (df['WEIGHT']>=0) & (df['HEATTYPE'] >= 0) & (df['HEATFUEL']>= 0 ) & (df['HOTWATER']>=0) & (df['ACPRIMARY']>=0) & (df['ACPRIMARY']>=0) & (df['ACSECNDRY']>=0) & (df['HINCP']>= 0)]

print("The highest income in the dataset is $",COdf['HINCP'].max(), "per year.")
print("The lowest income in the dataset is $", df['HINCP'].min(), "per year.")
df.dtypes

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


ss = StandardScaler() 
logit = LogisticRegression(random_state=0, solver='liblinear')

import matplotlib.pyplot as plt
import seaborn as sns
def col_histogram(df,column):
    plt.figure(figsize=(12,8))
    sns.set(style="darkgrid")
    sns.distplot(df[column].values)
    plt.tight_layout()
    plt.show()

features = ['MONOXIDE','CONTROL','HEATTYPE','HEATFUEL','HOTWATER','ACPRIMARY','ACSECNDRY','HOA','RENTCNTRL','HHMAR','HHGRAD','HHRACE','INTLANG']
for i in features:
  col_histogram(df,i)

detectorscount = COdf[(COdf['MONOXIDE']>= 0 )]
detectorscount = detectorscount.groupby('MONOXIDE').MONOXIDE.count()
fig,ax = plt.subplots(figsize = (12,8))
detectorscount.plot(kind='bar', color = '#4363d8')
plt.ylabel('Count of Households with Detectors')
plt.xticks(np.arange(2),('Yes','No'), rotation = 'horizontal')
plt.show()

HINCPModel = COdf[(COdf['MONOXIDE']>= 0 )&(COdf['HINCP']>= 0 )]
HINCPModel = HINCPModel[['MONOXIDE','HINCP']]
X = HINCPModel['HINCP'].values
y = HINCPModel['MONOXIDE'].values

le = LabelEncoder()
y = le.fit_transform(y)
from imblearn.over_sampling import RandomOverSampler
y = y.reshape(-1,1)
X = X.reshape(-1,1)

# RandomOverSampler (with random_state=0)
# Implement me
ros = RandomOverSampler(random_state=0)
X, y = ros.fit_sample(X, y)

pd.DataFrame(data=y, columns=['MONOXIDE'])['MONOXIDE'].value_counts()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)
X_train = ss.fit_transform(X_train.reshape(-1, 1))
X_test = ss.transform(X_test.reshape(-1,1))

logit_model = logit.fit(X_train,y_train)
y_predicted = logit_model.predict(X_test)
accuracy_score(y_test, y_predicted)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print(confusion_matrix(y_test, y_predicted))
print(classification_report(y_test,y_predicted))

df_model1 = COdf[(COdf['MONOXIDE']>= 0 ) & (COdf['HOA']>=0) & (COdf['YRBUILT']>=0) & (COdf['RENT']>=0) & (COdf['RENTCNTRL']>=0) & (COdf['HHMAR']>=0) & (COdf['HHGRAD']>=0) & (COdf['HHRACE']>=0) & (COdf['INTLANG']>=0) & (COdf['WEIGHT']>=0) & (COdf['HEATTYPE'] >= 0) & (COdf['HEATFUEL']>= 0 ) & (COdf['HOTWATER']>=0) & (COdf['ACPRIMARY']>=0) & (COdf['ACPRIMARY']>=0) & (COdf['ACSECNDRY']>=0) & (COdf['HINCP']>= 0)]
df_model1 = df_model1.drop(columns = ["CONTROL", "OMB13CBSA", "WEIGHT"])
model1target = "MONOXIDE"
X1 = df_model1.drop(columns=[model1target]).values
y1 = df_model1[model1target].values

# RandomOverSampler (with random_state=0)
ros = RandomOverSampler(random_state=0)
X1, y1 = ros.fit_sample(X1, y1)

def cvf(pipe,X,y, n_splits):
    accs = cross_val_score(pipe,
                       X,
                       y,
                       cv=KFold(n_splits=n_splits, random_state=0))
    print('The average accuracy score of the model is ', round(accs.mean(), 3))
    print('The std deviation of the accuracy score is ', round(accs.std(), 3))
    return round(accs.mean(), 3)

pipe_logit = Pipeline([('StandardScaler', ss), ('Logistic', logit)])
pipe_rf = Pipeline([('StandardScaler',ss), ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=10, random_state=0))])
pipe_svm = Pipeline([('StandardScaler',ss), ('SVC', SVC(random_state=0))])

# prepare python dictionary with models to test
pipe_estimators_model1 = {}

pipe_estimators_model1['Logistic'] = pipe_logit
pipe_estimators_model1['Random Forest'] = pipe_rf
pipe_estimators_model1['SVC'] = pipe_svm

Kfolds = 5
scores = []

for name, pipe in pipe_estimators_model1.items():
    print('\n model {} ...'.format(name))
    #%time score = cvf(pipe, X1,y1,Kfolds)
    start_time = datetime.now()
    score = cvf(pipe, X1,y1,Kfolds)
    print(datetime.now()-start_time)
    scores.append((name, score))
    
print("\n ------------------------------")
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
print("accuracy  -  model with Kfolds={} cross-validation".format(Kfolds))

for score in sorted_scores:
    print("{:0.3f} - {}".format(score[1], score[0]))

Model1_dict = dict(zip(df_model1.drop(columns=[model1target]).columns, pipe_rf.steps[1][1].fit(X1,y1).feature_importances_))

SortModel1 = pd.Series(Model1_dict).sort_values(ascending=True)
print(SortModel1)
SortModel1 = SortModel1.reset_index()

print('According to our random forest model, the most important features are income, rent status, education level, heat type, and year built')
plt.xlabel("Relative Importance")
plt.ylabel('Features')
plt.yticks(range(len(list(SortModel1['index']))), list(SortModel1['index']))
ax.invert_yaxis()
importances=pd.Series(RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=10, random_state=0).fit(X1,y1).feature_importances_)
plt.barh(range(len(list(SortModel1['index']))), list(SortModel1[0]), align='center')
plt.show()

#We can also use logistic regression to see how the features change the likelihood of CO monitor presence in a household

Model1v2_dict = dict(zip(df_model1.drop(columns=[model1target]).columns, (pipe_logit.steps[1][1].fit(X1,y1).coef_).flat))
SortModel1v2 = pd.Series(Model1v2_dict).sort_values(ascending=True)
print(SortModel1v2)
print("As seen above, higher income, newer homes, lower rent, fuel-burning appliances and lower education levels increase the likelihood of a household with a CO monitor")

#Now lets see if ensembling improves the accuracy
from sklearn.ensemble import VotingClassifier
pipe_ensemble = Pipeline([('StandardScaler', ss), ('Vote', VotingClassifier(estimators=[('Logit', pipe_logit), ('rf', pipe_rf), ('svc', pipe_svm)], voting='hard'))])
pipe_estimators_model2 = {}
pipe_estimators_model2['Vote'] = pipe_ensemble

Kfolds = 5
scores = []

for name, pipe in pipe_estimators_model2.items():
    print('\n model {} ...'.format(name))
    #%time score = cvf(pipe, X1,y1,Kfolds)
    start_time=datetime.now()
    score = cvf(pipe, X1,y1,Kfolds)
    print(datetime.now() - start_time)
    scores.append((name, score))
    
print("\n ------------------------------")
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
print("accuracy  -  model with Kfolds={} cross-validation".format(Kfolds))

for score in sorted_scores:
    print("{:0.3f} - {}".format(score[1], score[0]))

print('Since the ensembling method did not improve accuracy, we will analyze the dataset using the models using only the five most important features')
df_model3 = COdf[(COdf['MONOXIDE']>= 0 ) & (COdf['YRBUILT']>=0) & (COdf['RENT']>=0) & (COdf['HHGRAD']>=0) & (COdf['HEATTYPE'] >= 0) & (COdf['HINCP']>= 0)]
df_model3 = df_model3.drop(columns = ["CONTROL", "OMB13CBSA", "WEIGHT"])
model3target = "MONOXIDE"
X3 = df_model3.drop(columns=[model3target]).values
y3 = df_model3[model3target].values

# RandomOverSampler (with random_state=0)
ros = RandomOverSampler(random_state=0)
X3, y3 = ros.fit_sample(X3, y3)

Kfolds = 5
scores = []

pipe_estimators_model3 = pipe_estimators_model1

for name, pipe in pipe_estimators_model3.items():
    print('\n model {} ...'.format(name))
    #%time score = cvf(pipe, X1,y1,Kfolds)
    start_time = datetime.now()
    score = cvf(pipe, X3,y3,Kfolds)
    print(datetime.now()-start_time)
    scores.append((name, score))
    
print("\n ------------------------------")
sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
print("accuracy  -  model with Kfolds={} cross-validation".format(Kfolds))

for score in sorted_scores:
    print("{:0.3f} - {}".format(score[1], score[0]))

