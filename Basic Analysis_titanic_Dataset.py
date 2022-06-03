#---------------------------IMPORTS--------------------------------------------
import numpy as np
import pandas as pd
import pylab
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as se
from scipy.stats import chi2_contingency
#-----------------------------import of files-------------------------------------
titanic=pd.read_csv("train.csv")
#------------------------------------Metadata-------------------
#Embarked implies where the traveler mounted from. There are three possible values for Embark — Southampton, Cherbourg, and Queenstown.
#Pclass is the class of titanic
#Sex male and Female
#SibSp refrs to the number of sibilings or spouse they have during the travel
#Parch refers to the number of children of parents passengers have

#--------------------------------Basic Cleanings------------------------------------
print("list of attributes",list(titanic.columns))
print("count",titanic.count())
#--------------------------------------------------------------
print("numenr of survived:",len(titanic[titanic.Survived==1]))
print("numebr of killed:", len(titanic[titanic.Survived==0]))
vals=set(titanic.Sex)
print(vals)
print("number of male:",len(titanic[titanic.Sex=="male"]))
print("number of female:",len(titanic[titanic.Sex=="female"]))
print("number of people with sibliling of spouse aboard",len(titanic[titanic.SibSp>=1]))
print("number of people with no sibliling of spouse aboard",len(titanic[titanic.SibSp==0]))
print("numebr of people with childeren or parents aboard",len(titanic[titanic.Parch>=1]))
print("numebr of people without childeren or parents aboard",len(titanic[titanic.Parch==0]))
print("number of passengers without Cabin number:",len(titanic.isnull().sum()))
print("Embark:",set(titanic.Embarked))
print("Number of people in class S",len(titanic[titanic.Embarked=="S"]))
print("Number of people in class C",len(titanic[titanic.Embarked=="C"]))
print("Number of people in class Q",len(titanic[titanic.Embarked=="Q"]))
print("Number of people in class nan",len(titanic.isnull().sum()))
#-------------------------General info-------------------------------------------
# print(titanic.info())
print(titanic.describe(include="all"))
# print(titanic["Age"].describe())
#------------------------------------Mean, Trimed mean------------------------
#for some columns we should calculate the Mean and Trimed mean
print(stats.trim_mean(titanic.Fare,0.1))
#-------------------------------------------------
titanic_number=titanic._get_numeric_data()
collist=list(titanic_number.columns)
#----------------------------median and mean----------------------
#Median
print("mean",titanic_number.mean())
print("median",titanic_number.median())
#-------------------------------outlier Detection--------------------

def outlier_detection(col,df):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    IQR = q3 - q1
    low = q1 - 1.5 * IQR
    high = q3 + 1.5 * IQR
    outliers=df.loc[(df[col]>high) | (df[col]<low)]
    return outliers
#----------------------------------------------------------
for col in collist:
    # print("outliers of ",col)
    outliers=outlier_detection(col,titanic)
    print(len(outliers))
    print(outliers)
#-------------------------------------------Variance and SD------------------------------------
print("skew",titanic_number.skew())
print("Variance",titanic_number.var())
print()
#-------------------------General Distributions---------------------------------

#----------Histogram-------------
collist.remove("PassengerId")
collist.append("Sex")

for col in collist:
    plt.hist(titanic[col])
    plt.title(col)
    plt.show()
    plt.savefig("hist_"+col)

#-----------------Box plot--------

#---- all attrs together in the same figure-------
titanic_number["Age"].fillna(titanic_number["Age"].mean())
melted_titanic=pd.melt(titanic_number)
se.boxplot(x="variable",y="value",data=melted_titanic)
plt.show()
#---------------------box plot of each atter individaully----------------
collist.remove("Sex")
for col in collist:
    print(col)
    plt.boxplot(titanic[col].fillna(titanic[col].mean()))
    plt.title(col)
    plt.savefig("box_"+col)
    plt.show()

#-----------------------------------------end of Boxplot---------------------------

#---------------Scatter Plot-----------
#turn age to categorical type
titanic.fillna(np.mean(titanic["Age"]))
category=pd.cut(titanic.Age,bins=[0,2,17,65,99],labels=['Toddler','Child','Adult','Elderly'])
titanic.insert(5,"Age group",category)
print(titanic["Age group"])
print(titanic["Age group"].value_counts(normalize=True))
#----------------
# Survived vs Sex
plt.clf()
se.scatterplot(data=titanic,x=titanic["Survived"],y=titanic["Age"],hue=titanic["Embarked"])
plt.savefig("Survived-Age")
plt.show()

# Survived vs Age
plt.clf()
se.histplot(data=titanic,x=titanic["Survived"],y=titanic['Age group'],hue=titanic["Survived"])
plt.savefig("Age group Survived")
plt.show()
#-------------------------------------
plt.clf()
se.scatterplot(data=titanic,x=titanic["Survived"],y=titanic["Age"],hue=titanic["Sex"])
plt.savefig("Survived-Age")
plt.show()


# Survived vs sib
plt.clf()
se.scatterplot(data=titanic,x=titanic["Survived"],y=titanic["SibSp"],hue=titanic["Sex"])
plt.savefig("Survived-SibSp")
plt.show()

# Survived vs parch
plt.clf()
se.scatterplot(data=titanic,x=titanic["Survived"],y=titanic["Parch"],hue=titanic["Sex"])
plt.savefig("surivived-Sex")
plt.show()
plt.clf()
se.histplot(data=titanic,x=titanic["Survived"],y=titanic["Parch"])
plt.savefig("hist_survived_Parch")
plt.show()

# Survived vs Embarked
plt.clf()
se.histplot(data=titanic,x=titanic["Survived"],y=titanic["Embarked"])
plt.savefig("Hist_Surv_Embarked")
plt.show()

# Pclass vs Sex
plt.clf()
se.histplot(data=titanic,x=titanic['Pclass'],y=titanic["Sex"])
plt.savefig("Hist_Pclass_Sex")
plt.show()


# Pclass vs Age
plt.clf()
se.histplot(data=titanic,x=titanic["Age group"],y=titanic["Pclass"])
plt.savefig("hist_Age_pclass")
plt.show()


# Pclass vs Embarked
plt.clf()
se.histplot(data=titanic,x=titanic["Pclass"],y=titanic["Embarked"])
plt.savefig("hsit_pclass_embarked")
plt.show()


# Sex vs Age
plt.clf()
se.histplot(data=titanic,x=titanic["Age group"],y=titanic["Sex"])
plt.savefig("bar_sex_age")
plt.show()
#--------------------------Correlation-------------------------------------------
print(list(titanic.columns))
#-----------------------Survived vs Sex---------------------
#return chi2: the test statistics, P the p=value of the test, dof: Degree of freedom,
# Expected: The expected frequencies, based on the marginal sums of the table.
crosstabresult=pd.crosstab(index=titanic["Survived"],columns=titanic["Sex"])
print(crosstabresult)
chiresult=chi2_contingency(crosstabresult)
print("chiresult",chiresult)
#--------------------Age group vs Survived----------------
# category=pd.cut(titanic.Age,bins=[0,2,17,65,99],labels=['Toddler','Child','Adult','Elderly'])
# titanic.insert(5,"Age group",category)
# print(titanic["Age group"].value_counts(normalize=True))
crosstabresult=pd.crosstab(index=titanic["Survived"],columns=titanic["Age group"])
print(crosstabresult)
chiresult=chi2_contingency(crosstabresult)
print(chiresult)
#----------------------Survived vs Pclass----------
crosstabresult=pd.crosstab(index=titanic["Survived"],columns=titanic["Pclass"])
print(crosstabresult)
chiresult=chi2_contingency(crosstabresult)
print(chiresult)
#---------------------------------

#-------------------------QQ plot for investigating the deviation from normal distribution-----------------
titanic.fillna(np.mean(titanic["Age"]))
stats.probplot(titanic["Age"],dist="norm",plot=pylab)
pylab.savefig("QQ_Age")
pylab.show()
#------------------------------------------------------------------------------------------