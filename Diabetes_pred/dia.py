from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn.svm import SVC

# Load the diabetes dataset
df=pd.read_csv('C:/Users/91861/Downloads/diabetes.csv')

data=df.drop(['Pregnancies','SkinThickness','DiabetesPedigreeFunction','Outcome'],axis=1)

x=data

y=df['Outcome']

regressor=LogisticRegression()
classifier=SVC()


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42)
x_train.shape,y_train.shape,x_test.shape,y_test.shape


regressor.fit(x,y)

classifier.fit(x,y)

reg_pred=regressor.predict(x_test)
class_pred=classifier.predict(x_test)

reg_mse=mean_squared_error(y_test,reg_pred)
class_acc=accuracy_score(y_test,class_pred)

print('Mean Squared error of Logistic regressor:',reg_mse)
print('Accuracy of Support Vector Classifier:',class_acc)


import pickle

pickle.dump(classifier,open('classifier.pkl','wb'))

model=pickle.load(open('classifier.pkl','rb'))
print('-->>The Person having diabetes :',model.predict([[120,70,150,20,32]]))