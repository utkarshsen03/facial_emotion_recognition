import os
# Image analysis
import cv2
# For Web-deployment
import streamlit as st
# Loading Image
from PIL import Image
# Numerical Analysis
import numpy as np
# Loading saved models
from keras.models import load_model
# Visualization
import plotly.express as px
# Datetime
from datetime import datetime
# Face detection
from uploads.face_recognition import preprocessing


@st.cache_resource
def model_use():
    '''returns: saved model'''
    model = load_model(r'models\model_1.h5')
    return model

@st.cache_data
def prediction(source):
    '''inputs:
    source: source of the image
    returns: probabilities of different classes'''
    img = Image.open(source)
    img = img.convert('RGB')
    img.save(r'data\img.jpg')
    image = preprocessing(r'data\img.jpg')
    face = image.cascade()
    image.reshape()
    output = image.standardize()
    model = model_use()
    pred = model.predict(output)
    return pred,face
        
def results(prediction):
    ''' Prints label with highest probability and pie chart with all probability distribution
    inputs:
    prediction: probabilities of different classes
    returns: label with highest probability'''
    data = np.round(prediction.flatten() * 100,decimals=2)
    labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
    emojis = [':rage:',':face_vomiting:',':fearful:',':smile:',':neutral_face:',':cry:',':exploding_head:']
    idx = np.argmax(data)
    result = labels[idx]
    st.title('You are feeling:')
    st.title(result + emojis[idx])
    with st.expander('Probabilites of different emotions'):
        fig = px.pie(values=data,names=labels,title='Probability of Emotions')
        st.plotly_chart(fig,use_container_width=True)
    return result

def check_prediction(result,face):
    '''saves the image in folder for future training.
    inputs:
    result:label with highest probability
    face: face image'''
    with st.expander('Help Model Improve'):
        check = st.radio('Is prediction correct?',['Yes','No'],index=None)
        dir = r'data\new_data'
        if check == 'Yes':
            result = result.lower()
            img_dir = os.path.join(dir,'correctly_classified',result)
            unique_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(img_dir,unique_name)
            cv2.imwrite(image_path+'.jpg',face)
            st.write('Thank you for the input.')
        if check == 'No':
            img_dir = os.path.join(dir,'incorrectly_classified')
            unique_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(img_dir,unique_name)
            cv2.imwrite(image_path+'.jpg',face)
            st.write('Thank you for the input.')
                     

# Title of the Web-page                     
st.title('Facial Emotion Recognition')
st.markdown('---')

# Sidebar
with st.sidebar:
    st.image(r'uploads\front_image.jpg')
    st.title("Facial Emotion Recognition")
    option = st.radio("**Select Image Capture Method :control_knobs: :** ",['Pre-loaded image','Web Camera','Upload Manually'],0)
    st.markdown('---')
    st.markdown('**Made By:** [Utkarsh Sen](https://www.linkedin.com/in/utk-sen/)')

with st.expander('**About Project** :question:'):
    st.write('**What this project is about?  :face_with_raised_eyebrow:**')
    st.info("""This app uses preloaded images, web camera, or manually uploaded images to capture your 
            face, analyzes your expression with clever algorithms, and guesses your emotion - happy, 
            sad, angry, surprised, and so on. It's like a mood detector for your face!""")
    st.write("**How to use this app? :thinking_face:**")
    st.warning("""To engage with the app, go to the sidebar and select a suitable method:
                1. If method chosen is pre-loaded images, use one of the 5 images to run model;
                2. If method chosen is web-camera, allow permission to use web camera and capture 
               the image; 3. If method chosen is upload manually, upload one of your image from the device.
               As a result, this would initiate the model process, display the model results as well 
               as allowing users to see a detailed pie-chart view of the probability of different classes.""")
    st.write('**Data Set Used:**')
    st.markdown('- [Face expression recognition dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)')
    
# Preloaded image    
if option == 'Pre-loaded image':
    dir = r'data\trial'
    paths = os.listdir(dir)
    file = st.selectbox('Choose a file:',paths,index=None,placeholder='choose a file')
    if file:
        image_path = os.path.join(dir,file)
        img = Image.open(image_path)
        st.image(img)        
        try:
            pred,face = prediction(image_path)
            result = results(pred)
        except:
            st.error("Please Import a Valid File.")

# Web Camera    
if option == 'Web Camera':
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        # To read image file buffer as a PIL Image:
        try:
            pred,face = prediction(img_file_buffer)
            result = results(pred)
            check_prediction(result,face)
        except:
            st.error("Please Import a Valid File.")

# User Images            
if option == 'Upload Manually':
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            st.image(img)
            pred,face = prediction(uploaded_file)
            result = results(pred)
            check_prediction(result,face)
        except:
            st.error("Please Import a Valid File.")