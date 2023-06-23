import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("C:/Users/yashb/Desktop/datascience intern/archive (2 boston )/HousingData.csv")
# print(df.isnull().sum())

# deal with Nan
df['CRIM'].fillna((df['CRIM'].mean()), inplace=True)
df['ZN'].fillna((df['ZN'].mean()), inplace=True)
df['INDUS'].fillna((df['INDUS'].mean()), inplace=True)
df['CHAS'].fillna((df['CHAS'].mean()), inplace=True)
df['AGE'].fillna((df['AGE'].mean()), inplace=True)
df['LSTAT'].fillna((df['LSTAT'].mean()), inplace=True)
# print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit(df["MEDV"])
df['MEDV']=le.transform(df['MEDV'])
print(df)
df=df.drop("PTRATIO",axis=1)
df=df.drop("INDUS",axis=1)
df=df.drop("NOX",axis=1)
df=df.drop("TAX",axis=1)
df=df.drop("B",axis=1)
x=df.drop("MEDV",axis=1)
y=df["MEDV"]

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier() #use to select important features
model.fit(x,y) #Take data and put in function/module
print("Importances:",model.feature_importances_) # important features are selected
feat_importance=pd.Series(model.feature_importances_,index=x.columns)
feat_importance.nlargest(10).plot(kind='barh')
plt.show()

from imblearn.over_sampling import RandomOverSampler
from collections import Counter
# print("Count:",Counter(y))
sms = RandomOverSampler(random_state=0)
x,y = sms.fit_resample(x,y)
# print(Counter(y))

from matplotlib import pyplot as plt
import seaborn as sns
sns.boxplot(df['DIS']) # for LSTAT, RM, CRIM, DIS
plt.show()
a=[df['DIS'],df['LSTAT'],df['RM'],df['CRIM']]
for i in a:
    print(i)
    Q1=i.quantile(0.25)
    Q3=i.quantile(0.75)
    IQR=Q3-Q1
    print("IQR:",IQR)
    upper=Q3+1.5*IQR
    lower=Q1-1.5*IQR
    print(upper)
    print(lower)
    out1=df[i<lower].values
    out2=df[i>upper].values
    i.replace(out1,lower,inplace=True)
    i.replace(out2,upper,inplace=True)
    sns.boxplot(i)
    plt.show()

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# logr=LogisticRegression(max_iter=500)
rfc=RandomForestClassifier()
dtc=DecisionTreeClassifier()
gbc=GradientBoostingClassifier()
# pca=PCA(n_components=6)
# pca.fit(x)
# x=pca.transform(x)
# print(x)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=0,test_size=0.39)

func=[rfc,dtc,gbc]

for item in func:
    item.fit(xtrain,ytrain)
    ypred=item.predict(xtest)
    h=(accuracy_score(ytest,ypred))
    print(item,h*100,"%")
