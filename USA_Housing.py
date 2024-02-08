import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import warnings
warnings.filterwarnings ('ignore')
import streamlit as st 
import joblib
from sklearn.linear_model import LinearRegression

data = pd.read_csv('USA_Housing.csv')

#import the model
model = joblib.load('Linear Regression Model.pkl')

st.markdown("<h1 style = 'color: #1F4172; text-align: center; font-family: helvetica '>HOUSING PRICE PREDICTION</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By Oluyemi Isaiah</h4>", unsafe_allow_html = True)


st.image('pngwing.com.png', width = 350, use_column_width = True )
st.markdown("<h1 style = 'color: #1F4172; text-align: center; font-family: helvetica '>Project Overview</h1>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<p>Discover your dream home with Ziyah and co, your trusted partner in finding the perfect property. Our experienced team specializes in matching clients with homes that suit their lifestyle and preferences. With a diverse portfolio of listings, personalized service, and a commitment to excellence, we make your real estate journey seamless. Elevate your living experience with us, where your dream home becomes a reality.</p>", unsafe_allow_html=True)     

st.sidebar.write('Feature Input')
st.markdown("<br>", unsafe_allow_html = True)
st.dataframe(data, use_container_width = True )

st.sidebar.image('pngwing.com (2).png', caption= 'Welcome User')



input_choice= st.sidebar.radio('Choose Your Input Type', ['Slider Input', 'Number Input'])
if input_choice== 'Slider Input':
    area_income= st.sidebar.slider('Average Area Income', data['Avg. Area Income'].min(), data['Avg. Area Income'].max())
    house_age= st.sidebar.slider('Average Area House Age', data['Avg. Area House Age'].min(), data['Avg. Area House Age'].max())
    room_num= st.sidebar.slider('Average Area Number of Rooms', data['Avg. Area Number of Rooms'].min(), data['Avg. Area Number of Rooms'].max())
    bedrooms= st.sidebar.slider('Average Area Number of Bedrooms', data['Avg. Area Number of Rooms'].min(), data['Avg. Area Number of Bedrooms'].max())
    area_population= st.sidebar.slider('Area Population', data['Area Population'].min(), data['Area Population'].max())
else:
    area_income= st.sidebar.number_input('Average Area Income', data['Avg. Area Income'].min(), data['Avg. Area Income'].max())
    house_age= st.sidebar.number_input('Average Area House Age', data['Avg. Area House Age'].min(), data['Avg. Area House Age'].max())
    room_num= st.sidebar.number_input('Average Area Number of Rooms', data['Avg. Area Number of Rooms'].min(), data['Avg. Area Number of Rooms'].max())
    bedrooms= st.sidebar.number_input('Average Area Number of Bedrooms', data['Avg. Area Number of Rooms'].min(), data['Avg. Area Number of Bedrooms'].max())
    area_population= st.sidebar.number_input('Area Population', data['Area Population'].min(), data['Area Population'].max())


input_var = pd.DataFrame({'Avg. Area Income': [area_income],
                           'Avg. Area House Age': [house_age], 
                           'Avg. Area Number of Rooms': [room_num],
                          'Avg. Area Number of Bedrooms':[bedrooms],
                           'Area Population':[area_population] })
# st.markdown(css + "<hr class= 'colorful-divider>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<h5 style= 'margin: -30px; color:olive; font:sans serif' >", unsafe_allow_html= True)
st.dataframe(input_var)

predicted = model.predict(input_var)
prediction, interprete = st.tabs(["Model Prediction", "Model Interpretation"])
with prediction:
    pred = st.button('Push To Predict')
    if pred: 
        st.success(f'The Predicted price of your house is {predicted}')

with interprete:
    st.header('The Interpretation Of The Model')
    st.write(f'The intercept of the model is: {round(model.intercept_, 2)}')
    st.write(f'A unit change in the average area income causes the price to change by {model.coef_[0]} naira')
    st.write(f'A unit change in the average house age causes the price to change by {model.coef_[1]} naira')
    st.write(f'A unit change in the average number of rooms causes the price to change by {model.coef_[2]} naira')
    st.write(f'A unit change in the average number of bedrooms causes the price to change by {model.coef_[3]} naira')
    st.write(f'A unit change in the average number of populatioin causes the price to change by {model.coef_[4]} naira')