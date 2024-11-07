from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
import pickle
from keras.models import load_model
from keras.applications import VGG16
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.models import model_from_json
from tensorflow.keras.layers import *
import random

import tensorflow as tf
from PIL import Image
from DataReader import DataReader


main = tkinter.Tk()
main.title("DFR-TSD: A Deep Learning Based Framework for Robust Traffic Sign Detection Under Challenging Weather Conditions")
main.geometry("1300x1200")

global filename
global model
global dehaze_model, saver, RGB, MAX

class_labels = ['Speed limit (20km/h)','Speed limit (30km/h)','Speed limit (50km/h)','Speed limit (60km/h)','Speed limit (70km/h)',
                'Speed limit (80km/h)','End of speed limit (80km/h)','Speed limit (100km/h)','Speed limit (120km/h)','No passing','Stop','No Entry',
                'General caution','Traffic signals']

#create CNN Encoder Decoder model for noise blur remove
def generateEncoderModel(RGB):
    cnn1 = Conv2D(3,1,1,padding="same",activation="relu",use_bias=True,kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(RGB)#layer 1 for spectral original images
    cnn2 = Conv2D(3,3,1,padding="same",activation="relu",use_bias=True,kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(cnn1)#layer 2 for degraded images
    encoder1 = tf.concat([cnn1,cnn2],axis=-1) #concatenate both original and degraded images as input
    cnn3 = Conv2D(3,5,1,padding="same",activation="relu",use_bias=True,kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(encoder1)
    encoder2 = tf.concat([cnn2,cnn3],axis=-1)#concatenate layer2 and layer3 to further filter images
    cnn4 = Conv2D(3,7,1,padding="same",activation="relu",use_bias=True,kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(encoder2)
    decoder = tf.concat([cnn1,cnn2,cnn3,cnn4],axis=-1) #decoder output reconstruct new clear image
    cnn5 = Conv2D(3,3,1,padding="same",activation="relu",use_bias=True,kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                   kernel_regularizer=tf.keras.regularizers.l2(1e-4))(decoder)
    MAX = cnn5 #max layer
    dehaze_model = ReLU(max_value=1.0)(tf.math.multiply(MAX,RGB) - MAX + 1.0) #replace pixels intensity
    return dehaze_model

def ImageClearModel():
    global dehaze_model, saver, RGB, MAX
    dr = DataReader()  #class to read training images
    tf.reset_default_graph() #reset tensorflow graph
    trainImages, testImages = dr.readImages("./Dataset/data/clear","./Dataset/data/haze") #reading normal and noisy image to generate tensorflow CNN object
    trainData, testData, itr = dr.generateTrainTestImages(trainImages,testImages) 
    next_element = itr.get_next()

    RGB = tf.placeholder(shape=(None,480, 640,3),dtype=tf.float32)
    MAX = tf.placeholder(shape=(None,480, 640,3),dtype=tf.float32)
    dehaze_model = generateEncoderModel(RGB) #loading and generating model

    trainingLoss = tf.reduce_mean(tf.square(dehaze_model-MAX)) #optimizations
    optimizerRate = tf.train.AdamOptimizer(1e-4)
    trainVariables = tf.trainable_variables()
    gradient = optimizerRate.compute_gradients(trainingLoss,trainVariables)
    clippedGradients = [(tf.clip_by_norm(gradients,0.1),var1) for gradients,var1 in gradient]
    optimize = optimizerRate.apply_gradients(gradient)

    saver = tf.train.Saver()

def loadModel():
    global model
    model = load_model('model/model.h5')
    pathlabel.config(text="CNN Traffic Sign Detection Model Loaded")
    text.delete('1.0', END)
    text.insert(END,"CNN Traffic Sign Detection Model Loaded\n\n");

def clearImage():
    global model 
    filename = filedialog.askopenfilename(initialdir="testImages")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    with tf.Session() as session:
        saver.restore(session,'./model/data')
        img = Image.open(filename)
        img = img.resize((640, 480))
        img = np.asarray(img) / 255.0
        img = img.reshape((1,) + img.shape)
        clear_Image = session.run(dehaze_model,feed_dict={RGB:img,MAX:img})
        dehaze = clear_Image[0]
        orig = cv2.imread(filename)
        orig = cv2.resize(orig, (400,364))
        clear_Image = clear_Image[0]*255
        clear_Image = cv2.resize(clear_Image, (480,364))
        clear_Image = cv2.cvtColor(clear_Image, cv2.COLOR_BGR2RGB)        
        cv2.imwrite("test.png", clear_Image)
        figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(10,6))
        axis[0].set_title("Hazy Image")
        axis[1].set_title("Haze Cleared Image")
        axis[0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        axis[1].imshow(dehaze)
        figure.tight_layout()
        plt.show()

def detectSign():
    global model
    model = load_model('model/model.h5')
    text.delete('1.0', END)
    output = cv2.imread("test.png")
    h, w, c = output.shape
    image = load_img("test.png", target_size=(80, 80))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    (boxPreds, labelPreds) = model.predict(image)
    boxPreds = boxPreds[0]
    print(boxPreds)
    startX = int((boxPreds[0] * w))
    startY = int((boxPreds[1] * h))
    endX = int((boxPreds[2] * w))
    endY = int((boxPreds[3] * h))
    startX = random.randint(startX, startX + 100)
    endY = endY - 50
    predict= np.argmax(labelPreds, axis=1)
    predict = predict[0]
    accuracy = np.amax(labelPreds, axis=1)
    print(str(class_labels[predict])+" "+str(accuracy)+" "+str(startX)+" "+str(startY)+" "+str(endX)+" "+str(endY))
    cv2.rectangle(output, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.imshow("output", output)
    cv2.waitKey(0)
    

def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    accuracy = data['val_class_label_accuracy']
    loss = data['val_class_label_loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Training Poch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.plot(loss, 'ro-', color = 'blue')
    plt.legend(['Propose CNN Accuracy', 'Propose CNN Loss'], loc='upper left')
    plt.title('Propose CNN Training Accuracy & Graph')
    plt.show()
    
    
font = ('times', 16, 'bold')
title = Label(main, text='DFR-TSD: A Deep Learning Based Framework for Robust Traffic Sign Detection Under Challenging Weather Conditions',anchor=W, justify=CENTER)
title.config(bg='yellow4', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 13, 'bold')
upload = Button(main, text="Generate & Load Traffic Sign CNN Model", command=loadModel)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)

markovButton = Button(main, text="Upload Test Image & Clear", command=clearImage)
markovButton.place(x=50,y=200)
markovButton.config(font=font1)

predictButton = Button(main, text="Detect Sign from Clear Image", command=detectSign)
predictButton.place(x=50,y=250)
predictButton.config(font=font1)

graphButton = Button(main, text="Propose CNN Training Graph", command=graph)
graphButton.place(x=50,y=300)
graphButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=78)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)

ImageClearModel()
main.config(bg='magenta3')
main.mainloop()
