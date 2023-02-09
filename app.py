import streamlit as st
import pickle
import numpy as np


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.pixabay.com/photo/2016/02/17/15/37/laptop-1205256_960_720.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 
# import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor")

# brand
company = st.selectbox('Brand_of_Laptop', df['Company'].unique())

# Type of Laptop
type1 = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('Ram(in GB)', [2,4,6,8,12,16,24,32,64])

# weight of the laptop
weight = st.number_input('Weight of the Laptop')

# Touchscreen

touchscreen = st.selectbox('TouchScreen', ['No','Yes'])

# IPS
ips = st.selectbox('IPS', ['No','Yes'])

# screen size
screen_size = st.number_input('Screen Size')

# Resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2568x1440','2304x1440'])

# CPU
cpu = st.selectbox('CPU Brand', df['Cpu Brand'].unique())

# HHD
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

# SSD
ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

#GPU brand
gpu = st.selectbox('GPU',df['Gpu Brand'].unique())

# OS
os = st.selectbox('OS',df['OS'].unique())

if st.button('Predict Price'):
    #query
    ppi = None

    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])

    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,type1,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

  # 1 row 12 columns

    query = query.reshape(1,12)
    st.title("The Predicted Price for this configuration is ")
    st.title(np.exp(pipe.predict(query))[0])