import cv2 
import os 
import numpy as np 
from random import shuffle 
from tqdm import tqdm

import streamlit as st 
from PIL import Image



st.title("Fruit Prediction")

st.sidebar.title("About")

st.sidebar.info(
    "Our Project is the classification of fruits(namely Apple and Banana) using Convolutional Neural Networks. The Convolutional Neural Network (CNN) is a class of deep learning neural networks.")
st.sidebar.info(
        "CNNs represent a huge breakthrough in image recognition. They're most commonly used to analyze visual imagery and are frequently working behind the scenes in image classification.")

  
'''Setting up the envirnoment'''
  
TRAIN_DIR = 'fruit_train'
TEST_DIR = 'fruit_test'
IMG_SIZE = 50
LR = 1e-3
  
  
'''Setting up the model which will help with tensorflow models'''
MODEL_NAME = 'applev/sbanana-{}-{}.model'.format(LR, '6conv-basic') 
  
def label_img(img):
    word_label = img.split('.')[0][:5]
    if word_label == 'apple': return [1, 0] 
    elif word_label == 'banan': return [0, 1] 
  

def create_train_data(): 
    # Creating an empty list where we should store the training data 
    # after a little preprocessing of the data 
    training_data = [] 
  
    # tqdm is only used for interactive loading 
    # loading the training data 
    for img in tqdm(os.listdir(TRAIN_DIR)): 
  
        # labeling the images 
        label = label_img(img) 
  
        path = os.path.join(TRAIN_DIR, img) 
  
        # loading the image from the path and then converting them into 
        # greyscale for easier covnet prob 
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
  
        # resizing the image for processing them in the covnet 
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) 
  
        # final step-forming the training data list with numpy array of the images 
        training_data.append([np.array(img), np.array(label)]) 
  
    # shuffling of the training data to preserve the random state of our data 
    shuffle(training_data) 
  
    # saving our trained data for further uses if required 
    np.save('train_data.npy', training_data) 
    return training_data 
  
# Almost same as processing the training data but 
# we dont have to label it. 
def process_test_data(): 
    testing_data = [] 
    for img in tqdm(os.listdir(TEST_DIR)): 
        path = os.path.join(TEST_DIR, img) 
        img_num = img.split('.')[0] 
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) 
        testing_data.append([np.array(img), img_num]) 
          
    shuffle(testing_data) 
    np.save('test_data.npy', testing_data) 
    return testing_data 


  
'''Running the training and the testing in the dataset for our model'''
train_data = create_train_data() 
test_data = process_test_data() 

'''Creating the neural network using tensorflow'''
# Importing the required libraries 
import tflearn 
from tflearn.layers.conv import conv_2d, max_pool_2d 
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression 
  
import tensorflow as tf 
tf.reset_default_graph() 
convnet = input_data(shape =[None, IMG_SIZE, IMG_SIZE, 1], name ='input') 
  
convnet = conv_2d(convnet, 32, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 64, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 128, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 64, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = conv_2d(convnet, 32, 5, activation ='relu') 
convnet = max_pool_2d(convnet, 5) 
  
convnet = fully_connected(convnet, 1024, activation ='relu') 
convnet = dropout(convnet, 0.8) 
  
convnet = fully_connected(convnet, 2, activation ='softmax') 
convnet = regression(convnet, optimizer ='adam', learning_rate = LR, 
      loss ='categorical_crossentropy', name ='targets') 
  
model = tflearn.DNN(convnet, tensorboard_dir ='log') 
  
# Splitting the testing data and training data 
train = train_data[:-500] 
test = train_data[-500:] 
  
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
Y = [i[1] for i in train] 
test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 
test_y = [i[1] for i in test] 
  
'''Fitting the data into our model'''
# epoch = 5 taken 
model.fit({'input': X}, {'targets': Y}, n_epoch = 5,  
    validation_set =({'input': test_x}, {'targets': test_y}),  
    snapshot_step = 50, show_metric = True, run_id = MODEL_NAME) 
model.save(MODEL_NAME) 
  

import matplotlib.pyplot as plt 
test_data = np.load('test_data.npy',allow_pickle=True) 
  
fig = plt.figure() 
  
for num, data in enumerate(test_data[:20]): 
    img_num = data[1] 
    img_data = data[0] 
      
    y = fig.add_subplot(4, 5, num + 1) 
    orig = img_data 
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1) 
  
    model_out = model.predict([data])[0] 
      
    if np.argmax(model_out) == 1: str_label ='Banana'
    else: str_label ='Apple'
          
    y.imshow(orig, cmap='gray') 
    plt.title(str_label) 
    y.axes.get_xaxis().set_visible(False) 
    y.axes.get_yaxis().set_visible(False) 
plt.show()



st.subheader(" Lets Predict ")

def predict_test_data(image): 
    image = cv2.imread(ph, cv2.IMREAD_GRAYSCALE) 
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) 
    return image


uploaded_file = st.file_uploader("Choose an image...", type="jpg")
ph="banana.jpg"

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    imm1=predict_test_data(image)
    
    
    data = imm1.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]
    st.write("Classifying...............")
    st.write("Classified")
    if np.argmax(model_out) == 0: str_label ='Apple'
    else: str_label ='Banana'
    st.subheader('%s ' % (str_label))
