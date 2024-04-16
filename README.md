# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics. 10.Find the accuracy of our model and predict the require values.

Program:
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SIBIRAJ E
RegisterNumber:  212223080052
*/
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Employee (1).csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

![image](https://github.com/Sibi-raj5/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/160597836/881c91dd-e1b0-43c4-8e8c-c02819e7af2f)

![image](https://github.com/Sibi-raj5/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/160597836/048f14a5-56cd-4879-8696-c87db2bff1ff)

![image](https://github.com/Sibi-raj5/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/160597836/e81e418a-c91b-41e2-a0da-ba09836993c2)

![image](https://github.com/Sibi-raj5/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/160597836/0b10b933-f019-4c1b-888a-3a0d86b767e3)

![image](https://github.com/Sibi-raj5/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/160597836/24c48dc3-7c7e-4162-84e4-d97b69bd0569)

![image](https://github.com/Sibi-raj5/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/160597836/10d1a7aa-68ff-434e-9fd3-cb035be62323)

![image](https://github.com/Sibi-raj5/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/160597836/5c88de7e-879f-4c53-b2d3-780a8ac6a7d5)









## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
