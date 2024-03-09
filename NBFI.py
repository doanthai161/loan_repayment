import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
np.random.seed(0)

train = pd.read_csv('dataset/loan repayment/Train_dataset.csv',sep = ',')
test = pd.read_csv('dataset/loan repayment/Test_dataset.csv',sep = ',')


# print(train.head())
# print(train.describe())
# print(train.isnull().sum().sort_values(ascending=False))
train = train.drop(['Own_House_Age', 'Score_Source_1', 'Social_Circle_Default', 'Client_Occupation', 'Score_Source_3'], axis = 1)
# print(test.isnull().sum().sort_values(ascending=False))
test = test.drop(['Own_House_Age', 'Score_Source_1', 'Social_Circle_Default', 'Client_Occupation', 'Score_Source_3'], axis = 1)

#print(train.info())

def clean(df):
    type_invalid=['#', '@', 'x', '$', '#VALUE!']
    df.replace(type_invalid,None, inplace = True)

    name_col = [
        'Credit_Bureau', 'Client_Income_Type', 'Bike_Owned', 'Type_Organization', 'Active_Loan',
        'Accompany_Client', 'Client_Marital_Status', 'Client_Housing_Type', 'Application_Process_Hour',
        'Car_Owned', 'Client_Education', 'House_Own', 'Loan_Contract_Type', 'Cleint_City_Rating',
        'Client_Gender', 'Application_Process_Day', 'ID_Days', 'Score_Source_2', 'Population_Region_Relative',
        'Loan_Annuity', 'Age_Days', 'Client_Income', 'Phone_Change', 'Employed_Days', 'Registration_Days',
        'Child_Count', 'Credit_Amount', 'Client_Family_Members'
    ]
    for col in name_col:
        df[col] = df[col].fillna(df[col].mode()[0] if df[col].dtype == 'object' else df[col].median())

    return df

train = clean(train)
test = clean(test)

# print(train.isnull().sum().sort_values(ascending=False))

# plt.figure(figsize = (25, 20))
# plt.suptitle("Analysis Of Variable Default",fontweight="bold", fontsize=20)

# plt.subplot(5,2,1)
# sns.boxplot(x="Default", y="Client_Income", data=train)

def group(value):
    if value >= 0.8:
        return 0
    elif value >= 0.7  and value < 0.8:
        return 1
    elif value >= 0.6  and value < 0.7:
        return 2
    elif value >= 0.5  and value < 0.6:
        return 3
    elif value >= 0.4  and value < 0.5:
        return 4
    elif value >= 0.3  and value < 0.4:
        return 5
    elif value >= 0.2  and value < 0.3:
        return 6
    elif value >= 0.1  and value < 0.2:
        return 7
    else:
        return 8

train['Score_Source_2']=train['Score_Source_2'].astype('float')
test['Score_Source_2']=test['Score_Source_2'].astype('float')

# print(train['Score_Source_2'].dtypes)

train['score_group'] = train.apply(lambda x: group(x['Score_Source_2']),axis=1)
train = train.drop('Score_Source_2', axis = 1)

test['score_group'] = test.apply(lambda y: group(y['Score_Source_2']),axis=1)
df = test.drop('Score_Source_2', axis = 1)

# print(train['score_group'].value_counts)

train = train.drop('ID', axis=1)
test = test.drop('ID', axis=1)

from sklearn.preprocessing import OneHotEncoder

def onehot(one):
    one = one[one['Client_Gender'] != 'XNA']
    one = one[one['Mobile_Tag'] == 1]
    hot = pd.get_dummies(one[[
        'Accompany_Client', 'Client_Income_Type', 'Client_Education', 'Client_Marital_Status',
        'Client_Gender', 'Loan_Contract_Type', 'Client_Housing_Type', 'Application_Process_Day', 
        'Client_Permanent_Match_Tag', 'Client_Contact_Work_Tag', 'Type_Organization'
        ]])
    
    one = pd.concat([one, hot], axis = 1)
    one = one.drop([
        'Accompany_Client', 'Client_Income_Type', 'Client_Education',
        'Client_Marital_Status', 'Client_Gender', 'Loan_Contract_Type', 'Client_Housing_Type',
        'Application_Process_Day', 'Client_Permanent_Match_Tag', 'Client_Contact_Work_Tag', 'Type_Organization'
        ], axis = 1)

    return one

train = onehot(train)
test = onehot(test)

X_train = train.values[:, :-1]
y_train = train.values[:, -1]
X_test = test.values[:, :-1]
y_test = test.values[:, -1]

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_standard = scaler.fit_transform(X_train)
X_test_standard = scaler.transform(X_test)

# print(X_train.shape)
# print(y_train.shape)


# from xgboost.sklearn import XGBClassifier

# parameters = {'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.8],
#               'max_depth': [1 , 2, 3, 4, 5, 6, 7, 8, 9],
#               'min_child_weight': [1, 3, 5, 7, 9],
#               'subsample': [0.1, 0.3, 0.5, 0.7, 0.9],
#               'colsample_bytree': [0.1, 0.3, 0.5, 0.7, 0.9],
#               'n_estimators': [500],
#               'gamma': [0.1, 0.3, 0.5, 0.7, 0.9],
#               'reg_alpha': [0.1, 0.3, 0.5, 0.7, 0.9],
#               'reg_lambda': [0.1, 0.3, 0.5, 0.7, 0.9]
#              }

# model = XGBClassifier()
# xgb_grid = RandomizedSearchCV(model,parameters, cv = 2, n_jobs = -1)
# xgb_grid.fit(X_train, y_train)

# print('Score: ', xgb_grid.best_score_)
# print('Params: ', xgb_grid.best_params_)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train_standard, y_train_encoded)

y_pred = dtc.predict(X_test_standard)

cf = confusion_matrix(y_test_encoded, y_pred)
print(cf)
acc = accuracy_score(y_test_encoded, y_pred)
print(acc)



