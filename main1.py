import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('titanic.csv')
df=df.dropna()
df.loc[df['Gender']=='male','label_sex']=0
df.loc[df['Gender']=='female','label_sex']=1
df.loc[df['Embarked']=='Southampton','label_embarked']=1
df.loc[df['Embarked']=='Queenstown','label_embarked']=2
df.loc[df['Embarked']=='Cherbourg','label_embarked']=3

X = df[['Pclass','label_sex','Age','label_embarked']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


def prediction(model, scaler):
    Pclass = int(input("Enter Class: "))
    label_sex = int(input("Enter Gender(enter 0 for male and 1 for female): "))
    Age = int(input("Enter Age: "))
    label_embarked = int(input("Enter Place Embarked(enter 1 for Southampton 2 for Queenstown and 3 for Cherbourg): "))
    
    user_input = np.array([Pclass, label_sex, label_sex, label_embarked])
    user_input_df = pd.DataFrame([user_input], columns=X.columns)
    
    user_input_scaled = scaler.transform(user_input_df)
    
    prediction= model.predict(user_input_scaled)
    
    if prediction[0]==1:
        print("Saved")
    else:
        print("Not Saved")

prediction(model, scaler)


