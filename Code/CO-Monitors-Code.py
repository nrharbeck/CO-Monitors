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

#Below is a logit model alone to see what sort of results we obtain related to the coefficients from earlier.
df_model2 = COdf[(COdf['MONOXIDE']>= 0 ) & (COdf['HOA']>=0) & (COdf['YRBUILT']>=0) & (COdf['RENT']>=0) & (COdf['RENTCNTRL']>=0) & (COdf['HHMAR']>=0) & (COdf['HHGRAD']>=0) & (COdf['HHRACE']>=0) & (COdf['INTLANG']>=0) & (COdf['WEIGHT']>=0) & (COdf['HEATTYPE'] >= 0) & (COdf['HEATFUEL']>= 0 ) & (COdf['HOTWATER']>=0) & (COdf['ACPRIMARY']>=0) & (COdf['ACPRIMARY']>=0) & (COdf['ACSECNDRY']>=0) & (COdf['HINCP']>= 0)]
df_model2 = df_model2.drop(columns = ["CONTROL", "OMB13CBSA", "WEIGHT"])
model2target = "MONOXIDE"
X2 = df_model2[['HOA','YRBUILT','RENT','RENTCNTRL','HHMAR','HHGRAD','HHRACE','INTLANG','HEATTYPE','HEATFUEL','HOTWATER','ACPRIMARY','ACSECNDRY','HINCP']].values
y2 = df_model2['MONOXIDE'].values
y2 = le.fit_transform(y2)

# RandomOverSampler (with random_state=0)
ros = RandomOverSampler(random_state=0)
X2, y2 = ros.fit_sample(X2, y2)

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.8, random_state=0)

LogitModel1 = logit.fit(X2_train,y2_train)

y2_predicted = logit.predict(X2_test)
print(confusion_matrix(y2_test, y2_predicted))
print(classification_report(y2_test,y2_predicted))

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

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.8, random_state=0)

rf_model3=pipe_rf.fit(X3_train, y3_train)

y3_rf_predicted=rf_model3.predict(X3_test)

rf_matrix=confusion_matrix(y3_test, y3_rf_predicted)

sns.heatmap(rf_matrix, annot=True, fmt="d")
plt.show()
print(classification_report(y3_test,y3_rf_predicted))

#GUI SHELL
from PyQt5.uic import loadUiType
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget


from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

Ui_MainWindow, QMainWindow = loadUiType('window.ui')


class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, ):
        super(Main, self).__init__()
        self.setupUi(self)
        self.fig_dict = {}

        self.mplfigs.itemClicked.connect(self.changefig)

        fig = Figure()
        self.addmpl(fig)

    def changefig(self, item):
        text = item.text()
        self.rmmpl()
        self.addmpl(self.fig_dict[text])

    def addfig(self, name, fig):
      self.fig_dict[name] = fig
      self.mplfigs.addItem(name)


    def addmpl(self, fig):
      self.canvas = FigureCanvas(fig)
      self.mplvl.addWidget(self.canvas)
      self.canvas.draw()
      self.toolbar = NavigationToolbar(self.canvas,
              self.mplwindow, coordinates=True)
      self.mplvl.addWidget(self.toolbar)

    def rmmpl(self,):
      self.mplvl.removeWidget(self.canvas)
      self.canvas.close()
      self.mplvl.removeWidget(self.toolbar)
      self.toolbar.close()

# Each of our main figures can be added to the GUI - named "Fig". Several examples included from the tutorial I used. Having trouble with the bar chart though.
if __name__ == '__main__':
  import sys
  from PyQt5 import QtGui
  import numpy as np

  fig1 = Figure()
  detectorscount = COdf[(COdf['MONOXIDE'] >= 0)]
  detectorscount = detectorscount.groupby('MONOXIDE').MONOXIDE.count()
  ax1f1 = fig1.add_subplot(111)
  ax1f1.bar(1, detectorscount[1])
  ax1f1.bar(2, detectorscount[2])
  ax1f1.set_xlabel('Owns a CO Monitor')
  ax1f1.set_ylabel('Count')
  ax1f1.set_xticks([1,2])
  ax1f1.set_xticklabels(['Yes','No'])
  fig1.suptitle('Households with CO Monitors')

  fig2 = Figure()
  ax1f2 = fig2.add_subplot(111)
  hoa = COdf[(COdf['HOA'] >= 0)]
  hoa = hoa.groupby('HOA').HOA.count()
  hoa = np.array(hoa)
  for i in range(len(hoa)):
      ax1f2.bar(i, hoa[i])
  ax1f2.set_xlabel('Is in a Homeowners Association')
  ax1f2.set_ylabel('Count')
  ax1f2.set_xticks([0,1])
  ax1f2.set_xticklabels(['Yes','No'])
  fig2.suptitle('HH: Homeowners Assc.')

  fig3 = Figure()
  ax1f3 = fig3.add_subplot(111)
  yrb = COdf[(COdf['YRBUILT'] >= 0)]
  yrb = yrb.groupby('YRBUILT').YRBUILT.count()
  yrb_lab = np.array(yrb.index)
  yrb = np.array(yrb)
  for i in range(len(yrb)):
      ax1f3.bar(i, yrb[i])
  ax1f3.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11])
  ax1f3.set_xticklabels(yrb_lab)
  ax1f3.set_xlabel('Year Home Built')
  ax1f3.set_ylabel('Count')
  fig3.suptitle('HH: Year Built.')

  fig4 = Figure()
  ax1f4 = fig4.add_subplot(111)
  ax1f4.set_xlabel('Monthly Rent Amount')
  ax1f4.set_ylabel('USD')
  rent = COdf[(COdf['RENT']) >=0]
  rent = rent.groupby('RENT').RENT.count()
  fig4.suptitle('HH: Rent')
  ax1f4.hist(rent)

  fig5 = Figure()
  ax1f5 = fig5.add_subplot(111)
  ax1f5.set_ylabel('Count')
  ax1f4.set_xlabel('Under Rent Control')
  rentctrl = COdf[(COdf['RENTCNTRL']) >=0]
  rentctrl = rentctrl.groupby('RENTCNTRL').RENTCNTRL.count()
  rentctrl = np.array(rentctrl)
  for i in range(len(rentctrl)):
      ax1f5.bar(i, rentctrl[i])
  ax1f5.set_xticks([0, 1])
  ax1f5.set_xticklabels(['Yes','No'])
  fig5.suptitle('HH: Rent Control')

  fig6 = Figure()
  ax1f6 = fig6.add_subplot(111)
  ax1f6.set_ylabel('Count')
  ax1f6.set_xlabel('Marital Status')
  hhmar = COdf[(COdf['HHMAR']) >=0]
  hhmar = hhmar.groupby('HHMAR').HHMAR.count()
  hhmar = np.array(hhmar)
  for i in range(len(hhmar)):
      ax1f6.bar(i, hhmar[i])
  ax1f6.set_xticks([0,1,2,3,4,5])
  ax1f6.set_xticklabels(['Married.p', 'Married.a','Widowed','Divorced','Separated','NM'])
  fig6.suptitle('HH: Marital Status')

  fig7 = Figure()
  ax1f7 = fig7.add_subplot(111)
  grad = COdf[(COdf['HHGRAD']) >=0]
  grad = grad.groupby('HHGRAD').HHGRAD.count()
  grad = np.array(grad)
  for i in range(len(grad)):
      ax1f7.bar(i, grad[i])
  ax1f7.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
  ax1f7.set_xticklabels(['<1st Gr.', '1-4 Gr.', '5-6 Gr.', '7-8 Gr.', '9 Gr.', '10th Gr.', '11 Gr.', '12 Gr.', 'HS/GED', 'Some College', 'Technical', 'Assoc./Voc.', 'Assc.', 'Bachelors', 'Masters','MD/JD/etc.', 'Doctorate'])
  fig7.suptitle('HH: Education Lvl')

  fig8 = Figure()
  ax1f8 = fig8.add_subplot(111)
  race = COdf[(COdf['HHRACE']) >= 0]
  race = race.groupby('HHRACE').HHRACE.count()
  race = np.array(race)
  r_other = np.array(sum(race[4:]))
  race = np.append(race[0:3], r_other) # NOTE: I COLLAPSED ALL RACES (low n) so this is WHITE, BLACK, NAT. AMER., AND ASIAN, + OTHERS.
  for i in range(len(race)):
      ax1f8.bar(i, race[i])
  fig8.suptitle('HH: Race')

  fig9 = Figure()
  ax1f9 = fig9.add_subplot(111)
  lang = COdf[(COdf['INTLANG']) >=0]
  lang = lang.groupby('INTLANG').INTLANG.count()
  lang = np.array(lang)
  for i in range(len(lang)):
      ax1f9.bar(i, lang[i])
  fig9.suptitle('HH: Language')

  fig10 = Figure()
  ax1f10 = fig10.add_subplot(111)
  heat = COdf[(COdf['HEATTYPE']) >=0]
  heat = heat.groupby('HEATTYPE').HEATTYPE.count()
  heat = np.array(heat)
  for i in range(len(heat)):
      ax1f10.bar(i, heat[i])
  fig10.suptitle('CO: Heat Type')

  fig11 = Figure()
  ax1f11 = fig11.add_subplot(111)
  heatfuel = COdf[(COdf['HEATFUEL']) >=0]
  heatfuel = heatfuel.groupby('HEATFUEL').HEATFUEL.count()
  heatfuel = np.array(heatfuel)
  for i in range(len(heatfuel)):
      ax1f11.bar(i, heatfuel[i])
  fig11.suptitle('CO: Heat Fuel')

  fig12 = Figure()
  ax1f12 = fig12.add_subplot(111)
  hotwater = COdf[(COdf['HOTWATER']) >=0]
  hotwater = hotwater.groupby('HOTWATER').HOTWATER.count()
  hotwater = np.array(hotwater)
  for i in range(len(hotwater)):
      ax1f12.bar(i, hotwater[i])
  fig12.suptitle('CO: Water Heater Type')

  fig13 = Figure()
  ax1f13 = fig13.add_subplot(111)
  acp = COdf[(COdf['ACPRIMARY']) >=0]
  acp = acp.groupby('ACPRIMARY').ACPRIMARY.count()
  acp = np.array(acp)
  for i in range(len(acp)):
      ax1f13.bar(i, acp[i])
  fig13.suptitle('CO: Primary Air Conditioning')

  fig14 = Figure()
  ax1f14 = fig14.add_subplot(111)
  acs = COdf[(COdf['ACSECNDRY']) >=0]
  acs = acs.groupby('ACSECNDRY').ACSECNDRY.count()
  acs = np.array(acs)
  for i in range(len(acs)):
      ax1f14.bar(i, acs[i])
  fig14.suptitle('CO: Secondary Air Conditioning')

  fig15 = Figure()
  ax1f15 = fig15.add_subplot(111)
  income = COdf[(COdf['HINCP']) >=0]
  income = income[['HINCP']]
  ax1f15.boxplot(income.values)
  fig15.suptitle('HH: Household Income (past 12 months)')

  app = QApplication(sys.argv)
  main = Main()
  main.addfig('Target: CO Monitor', fig1)
  main.addfig('HH: Homeowners Assc.', fig2)
  main.addfig('HH: Year Built', fig3)
  main.addfig('HH: Household Income', fig15)
  main.addfig('HH: Rent', fig4)
  main.addfig('HH: Rent Control', fig5)
  main.addfig('HH: Marital Status', fig6)
  main.addfig('HH: Education Lvl', fig7)
  main.addfig('HH: Race', fig8)
  main.addfig('HH: Language', fig9)
  main.addfig('CO: Heat Type', fig10)
  main.addfig('CO: Heat Fuel', fig11)
  main.addfig('CO: Water Heater', fig12)
  main.addfig('CO: AC Primary', fig13)
  main.addfig('CO: AC Secondary', fig14)
  main.show()
  sys.exit(app.exec_())
