import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

st.title("🚢 Titanic Survival Prediction")

df = pd.read_csv("train.csv")

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

df['FamilySize'] = df['SibSp'] + df['Parch']

df = df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1)

X = df.drop('Survived', axis=1)
y = df['Survived']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

st.header("Enter Passenger Details")

pclass = st.selectbox("Pclass", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouses", 0, 10, 0)
parch = st.number_input("Parents/Children", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 500.0, 50.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

sex = 1 if sex == "female" else 0
family = sibsp + parch

embarked_C = 1 if embarked == "C" else 0
embarked_Q = 1 if embarked == "Q" else 0

input_data = np.array([[pclass, sex, age, sibsp, parch, fare, family, embarked_C, embarked_Q]])

if st.button("Predict"):
    result = model.predict(input_data)
    if result[0] == 1:
        st.success("Survived 🎉")
    else:
        st.error("Did Not Survive 💀")