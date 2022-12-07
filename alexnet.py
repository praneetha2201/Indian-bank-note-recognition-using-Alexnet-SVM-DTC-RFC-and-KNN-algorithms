from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askdirectory
from tkinter import simpledialog
import cv2
from keras.utils.np_utils import to_categorical
from keras.layers import MaxPool2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.models import Sequential
import keras
import pickle
import matplotlib.pyplot as plt
import os
from keras.models import model_from_json
import alex

main = tkinter.Tk()
main.title("Fake Indian Currency Detection")
main.geometry("1000x700")


def upload():
  global filename
  filename = filedialog.askdirectory(initialdir = ".")
  alex.uploaddata(filename)
  text.delete('1.0', END)
  text.insert(END,filename+' Loaded')
  text.insert(END,'\n'+"Dataset Loaded")

  
def processImages():
    text.delete('1.0', END)
    alex.preprocessing()
    global X_train
    global Ytrain
    X_train = np.load('model/features.txt.npy')
    Ytrain = np.load('model/labels.txt.npy')
    text.insert(END,'Total images found in dataset for training = '+str(X_train.shape[0])+"\n\n")
    
  

def generateModel():
  global classifier
  text.delete('1.0', END)
  Y_train=to_categorical(Ytrain)
  classifier = Sequential()
  classifier.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3), padding='same'))
  classifier.add(BatchNormalization())
  classifier.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))
  classifier.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"))
  classifier.add(BatchNormalization())
  classifier.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))
  classifier.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
  classifier.add(BatchNormalization())
  classifier.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
  classifier.add(BatchNormalization())
  classifier.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
  classifier.add(BatchNormalization())
  classifier.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'))
  classifier.add(Flatten())
  classifier.add(Dense(4096, activation = 'relu'))
  classifier.add(BatchNormalization())
  classifier.add(Dropout(0.5))
  classifier.add(Dense(4096, activation = 'relu'))
  classifier.add(BatchNormalization())
  classifier.add(Dropout(0.5))
  classifier.add(Dense(1000, activation = 'relu'))
  classifier.add(BatchNormalization())
  classifier.add(Dropout(0.5))
  classifier.add(Dense(2, activation = 'softmax'))
  print(classifier.summary())
  classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
  hist = classifier.fit(X_train, Y_train, batch_size=36, epochs=20, shuffle=True, verbose=2)
  classifier.save_weights('model/model_weights.h5')            
  model_json = classifier.to_json()
  with open("model/model.json", "w") as json_file:
      json_file.write(model_json)
  f = open('model/history.pckl', 'wb')
  pickle.dump(hist.history, f)
  f.close()
  f = open('model/history.pckl', 'rb')
  data = pickle.load(f)
  f.close()
  acc = data['accuracy']
  accuracy = acc[19] * 100
  text.insert(END,"AlexNet Training Model Accuracy = "+str(accuracy)+"\n")
  
  
  
def predict():
    text.delete('1.0', END)
    name = filedialog.askopenfilename(initialdir="testImages")
    text.insert(END,'Test Image uploaded')
    img = cv2.imread(name)
    img = cv2.resize(img, (227,227))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,227,227,3)
    XX = np.asarray(im2arr)
    XX = XX.astype('float32')
    XX = XX/255
    preds = classifier.predict(XX)
    print(str(preds)+" "+str(np.argmax(preds)))
    predict = np.argmax(preds)
    print(predict)
    img = cv2.imread(name)
    img = cv2.resize(img,(450,450))
    msg = ''
    if predict == 0:
        cv2.putText(img, 'Fake', (100, 200),  cv2.FONT_HERSHEY_SIMPLEX,3, (255,0,0), 5)
        msg = 'Fake'
    else:
        cv2.putText(img, 'Real', (100, 200),  cv2.FONT_HERSHEY_SIMPLEX,3, (255,0,0), 5)
        msg = 'Real'
        
    cv2.imshow(msg,img)
    cv2.waitKey(0)

def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()

    accuracy = data['accuracy']
    loss = data['loss']

    plt.figure(figsize=(20,10))
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('AlexNet Accuracy & Loss')
    plt.show()
   
font = ('times', 16, 'bold')
title = Label(main, text='Detection of Fake Indian Currency ', justify=LEFT)
title.config(bg='yellow', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=upload)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Image Preprocessing", command=processImages)
processButton.place(x=280,y=100)
processButton.config(font=font1) 

cnnButton = Button(main, text="Generate Model", command=generateModel)
cnnButton.place(x=560,y=100)
cnnButton.config(font=font1) 

predictButton = Button(main, text="Upload Test Image", command=predict)
predictButton.place(x=10,y=150)
predictButton.config(font=font1)

graphButton = Button(main, text="Accuracy & Loss Graph", command=graph)
graphButton.place(x=280,y=150)
graphButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1) 

main.config(bg='LightSteelBlue3')
main.mainloop()
