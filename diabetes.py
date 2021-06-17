#import libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


#load the data
df=pd.read_csv('New folder/diabetes.csv')

X = df.iloc[:,1:8].values
Y=df.iloc[:,8].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.25 ,random_state =0)


#Scale the data (Feature scaling)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#create a function for the models
def models(X_train, Y_train):
    
    #logistic Regression
    from sklearn.linear_model import LogisticRegression
    log=LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)
    
    #Decision Tree classicifier
    from sklearn.tree import DecisionTreeClassifier
    tree=DecisionTreeClassifier(criterion= 'entropy', random_state=0)
    tree.fit(X_train,Y_train)
    
    #Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    forest= RandomForestClassifier(n_estimators = 10, criterion='entropy', random_state=0)
    forest.fit(X_train,Y_train)
    
    #Print the models Accuracy of the data
    print('[0] Logistic Regression training accuracy:',log.score(X_train, Y_train))
    print('[1] Decision Tree training accuracy:',tree.score(X_train, Y_train))
    print('[2] Random Forest Classifier training accuracy:',forest.score(X_train, Y_train))
    
    return log, tree,forest


st.write("# Diabetics Prediction Software")
st.write("  ")
images=open('diabetes-4948861_1920.jpg','rb').read()
st.image(images, width=400)


st.write("images downloaded from pixabay")



Pregnancies=st.text_input('How many month pregnant ? Leave it at 0 if you are not pregnant.') 

Glucose=st.text_input('What is the Glucose Level?')                  

BloodPressure=st.text_input('What is the blood pressure?')                  

SkinThickness=st.text_input('What is the skin thinkness?')                  

Insulin=st.text_input('What is the the level of Insulin?')                  

BMI=st.text_input('What is the BMI value?')                  

DiabetesPedigreeFunction=st.text_input('What is the Diabetes PredigreeFunction value?')                  

Age=st.text_input('Age of the person ?')  


arr={'Pregnancies':[Pregnancies],'Glucose':[Glucose],'BloodPressure':[BloodPressure],'SkinThickness':[SkinThickness],
     'Insulin':[Insulin],'BMI':[BMI],'DiabetesPedigreeFunction':[DiabetesPedigreeFunction],'Age':[Age]}

dg=pd.DataFrame(arr)

X = dg.iloc[:,0:7].values


model = models(X_train, Y_train)



if st.button("Make Prediction"): 
    st.write("Decision Tree training Prediction")
    sam=(model[2].predict(X))
    tom=accuracy_score(Y_test, model[2].predict(X_test))*100
    
    if sam == 1:
        st.write("You have   "+str(tom)+"     percent chance that you have Diabeties")
    else:
        st.write("You have    "+str(tom-100 )+"    percent chance that you are DO NOT HAVE Diabeties")


st.write("       ")

st.write("       ")

st.write("       ")

st.write("       ")


st.write("""
#  Context
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.



The datasets consist of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.


# Acknowledgements
Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). IEEE Computer Society Press.
Inspiration
Can you build a machine learning model to accurately predict whether or not the patients in the dataset have diabetes or not?


""")




