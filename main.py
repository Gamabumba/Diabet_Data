import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier


def user_report(): #Собираем данные от пользователя
    pregnancies = st.sidebar.slider('Перенесенных беременностей', 0, 10, 0)
    glucose = st.sidebar.slider('Уровень глюкозы', 0, 200, 120)
    bp = st.sidebar.slider('Давление', 0, 122, 70)
    skinthickness = st.sidebar.slider('Толщина кожи', 0, 100, 20)
    insulin = st.sidebar.slider('Инсулин', 0, 846, 79)
    bmi = st.sidebar.slider('Индекс массы тела', 0, 67, 20)
    dpf = st.sidebar.slider('Наследственность', 0.0, 2.4, 0.27)
    age = st.sidebar.slider('Возраст', 21, 88, 33)

    report = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    report = pd.DataFrame(report, index=[0])
    return report


df = pd.read_csv('diabetes.csv')

st.title('Определяем диабет')
st.subheader('Визуализация')
st.bar_chart(df)

X = df.drop(['Outcome'], axis=1)
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=True, shuffle=True)

user_data = user_report()

cat = CatBoostClassifier()
cat.fit(X_train, y_train)

st.subheader('Примерная точноть:')
st.write(str(accuracy_score(y_test, cat.predict(X_test)) * 100) + '%')

user_result = cat.predict(user_data)
st.subheader('Диагноз: ')
if user_result[0] == 0:
    output = 'Тебе повезло, диабета нет'
else:
    output = 'Готовь гроб, у тебя диабет'

st.write(output)
