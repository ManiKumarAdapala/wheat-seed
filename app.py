import streamlit as st
import numpy as np
from tensorflow import keras as kr

st.title("Wheat Seed Classifier")
st.write("- By Mani Kumar Adapala")

st.write("\n")

# Webpage

area = st.slider('Area :',10.59 ,21.18 ,14.8 )
st.write("\n")
perimeter = st.slider('Perimeter :', 12.41, 17.25, 14.5)
st.write("\n")
compactness = st.slider('Compactness :', 0.80, 0.91, 0.87)
st.write("\n")
kernelLength = st.slider('Kernel Length :', 4.89, 6.67, 5.62)
st.write("\n")
kernelWidth = st.slider('Kernel Width :', 2.63, 4.03, 3.25)
st.write("\n")
asymmetryCoefficient = st.slider('Asymmetry Coefficient :', 0.765, 8.456, 3.707)
st.write("\n")
lengthofKernelGroove = st.slider('Length of KernelGroove :', 4.51, 6.55, 5.40)
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
