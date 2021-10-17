import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras as kr

df = pd.read_csv("wheat-seeds.csv")
df.columns = ['Area','Perimeter','Compactness','LengthOfKernel','WidthOfKernel',
              'AsymmetryCoefficient','LengthOfKernelGroove','SeedType']

st.title("Wheat Seed Classifier")
st.write("- By Mani Kumar Adapala")

st.write("\n")

# Webpage

area = st.slider('Area :', df['Area'].min(), df['Area'].max(), df['Area'].mean())
st.write("\n")
perimeter = st.slider('Perimeter :', df['Perimeter'].min(), df['Perimeter'].max(), df['Perimeter'].mean())
st.write("\n")
compactness = st.slider('Compactness :', df['Compactness'].min(), df['Compactness'].max(), df['Compactness'].mean())
st.write("\n")
kernelLength = st.slider('Kernel Length :', df['LengthOfKernel'].min(), df['LengthOfKernel'].max(), df['LengthOfKernel'].mean())
st.write("\n")
kernelWidth = st.slider('Kernel Width :', df['WidthOfKernel'].min(), df['WidthOfKernel'].max(), df['WidthOfKernel'].mean())
st.write("\n")
asymmetryCoefficient = st.slider('Asymmetry Coefficient :', df['AsymmetryCoefficient'].min(), df['AsymmetryCoefficient'].max(), df['AsymmetryCoefficient'].mean())
st.write("\n")
lengthofKernelGroove = st.slider('Length of KernelGroove :', df['LengthOfKernelGroove'].min(), df['LengthOfKernelGroove'].max(), df['LengthOfKernelGroove'].mean())
st.write("\n")

# Model
model = kr.models.load_model("wheetseed_neural_network.h5")
out_vals = np.array([1,2,3])
pred = model.predict(np.expand_dims([area, perimeter, compactness, kernelLength, kernelWidth, asymmetryCoefficient, lengthofKernelGroove], axis=0))
out = np.around(pred).astype(np.int)
K = out_vals[out.astype(np.bool)[0]]

# [area, perimeter, compactness, kernelLength, kernelWidth, asymmetryCoefficient, lengthofKernelGroove]


if len(K) == 0 :
    st.write("Test is inconclusive. Kindly change More than 1 feature.")
else:
    if K[0] == 1 :
        st.write("Your Seed is Type 1")
        st.write("i.e, Kama Type Wheat Seed")
    elif K[0] == 2 :
        st.write("Your Seed is Type 2")
        st.write("i.e, Rosa Type Wheat Seed ")
    elif K[0] == 3 :
        st.write("Your Seed is Type 3")
        st.write("i.e, Canadian Type Wheat Seed ")

# links = '[Portfolio](https://manikumaradapala.gitlab.io/portfolio/) | [GitHub](https://github.com/ManiKumarAdapala) | [Linkedin](https://www.linkedin.com/in/manikumaradapala/)'
# st.markdown(links, unsafe_allow_html=True)

st.write("\n")
st.write("\n")
footer = f" <center> <br> <br> <br> <br> <h4> Explore more about me here.... </h4> </center>"
st.markdown(footer, unsafe_allow_html=True)

html = f"<center> <a href='https://www.linkedin.com/in/manikumaradapala'><img src='https://cdn-icons-png.flaticon.com/512/174/174857.png'  width='50' height='50'></a>  &emsp;  <a href='https://github.com/ManiKumarAdapala'><img src='https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/2048px-Octicons-mark-github.svg.png'  width='50' height='50'></a>   &emsp;  <a href='https://manikumaradapala.gitlab.io/portfolio/'><img src='https://cdn.iconscout.com/icon/premium/png-256-thumb/website-255-610491.png'  width='50' height='50'></a></center>"
st.markdown(html, unsafe_allow_html=True)

#st.write(f'Seed is "{K[0]}" ')
