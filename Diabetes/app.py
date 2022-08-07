from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from email.mime import image
from operator import index
import streamlit as st
import pandas as pd
import numpy as np
import string
import pickle
from PIL import Image
st.set_option('deprecation.showfileUploaderEncoding', False)
model = pickle.load(open('model.pkl', 'rb'))


st.title("Diabetes Predictor")


Pregnancies = st.slider("Input Your Number of Pregnancies", 0, 16)
Glucose = st.slider("Input your Gluclose", 74, 200)
BloodPressure = st.slider("Input your Blood Pressure", 30, 130)
SkinThickness = st.slider("Input your Skin thickness", 0, 100)
Insulin = st.slider("Input your Insulin", 0, 200)
BMI = st.slider("Input your BMI", 14.0, 60.0)
DiabetesPedigreeFunction = st.slider(
    "Input your Diabetes Pedigree Function", 0.0, 6.0)
Age = st.slider("Input your Age", 0, 100)


inputs = [[Pregnancies, Glucose, BloodPressure, SkinThickness,
           Insulin, BMI, DiabetesPedigreeFunction, Age]]
sad_image = Image.open('sad.jpeg')
happy_image= Image.open('happy.jpeg')


if st.button('Predict'):
    result = model.predict(inputs)
    updated_res = result.flatten().astype(int)
    if updated_res == 0:
        st.write("Sorry, You might have Diabetes. Take of Yourself")
        st.image(image=sad_image)
    else:
        st.write("You are Safe. But take care of your Health")
        st.image(image=happy_image)

st.title("Used Data-Set")
data = pd.read_csv("file.csv")
# st.write(data)


# Displaying the dataframe
gb = GridOptionsBuilder.from_dataframe(data)
gb.configure_pagination(paginationAutoPageSize=True)  # Add pagination
gb.configure_side_bar()  # Add a sidebar
gb.configure_selection('multiple', use_checkbox=True,
                       groupSelectsChildren="Group checkbox select children")  # Enable multi-row selection
gridOptions = gb.build()

grid_response = AgGrid(
    data,
    gridOptions=gridOptions,
    data_return_mode='AS_INPUT',
    update_mode='MODEL_CHANGED',
    fit_columns_on_grid_load=False,
    theme='blue',  # Add theme color to the table
    enable_enterprise_modules=True,
    height=350,
    width='100%',
    reload_data=True
)
data = grid_response['data']
selected = grid_response['selected_rows']
df = pd.DataFrame(selected)  # Pass the selected rows to a new dataframe df
