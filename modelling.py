import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from tabpy.tabpy_tools.client import Client

print('Initializing setup...')
data = pd.read_csv('bmi.csv')
model = None
client = Client("http://localhost:9004/")

def data_prep_n_modelling():
    print('Model Starting...')
    data['Gender'] = data['Gender'].replace({'Female':0,'Male':1})

    X = data.drop(columns='Index',axis=1)
    y = data['Index']
    train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=.3,random_state=8)

    model = LogisticRegression(multi_class='multinomial')

    model.fit(train_X,train_y)
    print(train_X.head(2))
    print('Model fitted...')
    print('...chk1...',model.predict([[1,174,96]]))
    '''
    pred_y = model.predict(test_X)
    print("Res:",classification_report(test_y,pred_y))
    print(model.predict([[1,174,96]]))
    joblib.dump(model,'model4rtabpy')
    '''
    return model

def getBMIstr(argip):
    swt = { 0:"Extreme Underweight", 1:"Underweight", 2:"Normal", 3:"Overweight", 4:"Obesity", 5:"Extreme Obesity"}
    return swt.get(argip)

def bmiClassifier(igender,iht,iwt):
    print('**********Deploy prediction cls...')
    print('disp:',igender,iht,iwt,[igender,iht,iwt])
    #conv_topred_data = pd.concat([igender,iht,iwt],axis=1)
    conv_topred_data = pd.DataFrame([[igender,iht,iwt]])
    bmi = int(model.predict(conv_topred_data))
    return getBMIstr(bmi)
    #return 199


print('Calling model prep...')
model = data_prep_n_modelling()

print('...chk2...',bmiClassifier(1,140,87))

print('Calling deploy...')
print(client.get_endpoints())
print('next')
client.deploy('bmiClassifier',bmiClassifier,'To predict BMI',override=True)
print('next2')
print(client.get_endpoints())
print('next3')
