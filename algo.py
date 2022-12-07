import numpy as np
import csv
import os
import cv2,glob
import matplotlib.pyplot as plt
from sklearn import metrics
from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askdirectory
from tkinter import simpledialog
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import classify

main = tkinter.Tk()
main.title("Indian currency classification")
main.geometry("1000x1000")

def upload():
  path = filedialog.askdirectory(initialdir = ".")
  classify.uploaddata(path)
  text.delete('1.0', END)
  text.insert(END,path+' Loaded\n')
  text.insert(END,"Dataset Loaded")

  
def processImages():
    text.delete('1.0', END)
    classify.preprocessing()
    global X_train
    global y_train
    global X_test
    global y_test
    X = np.load('model/features.txt.npy')
    Y = np.load('model/labels.txt.npy')
    X_train,X_test,y_train,y_test = train_test_split(X,Y, test_size = 0.2, random_state = 20)
    text.insert(END,'Total images found in dataset for training = '+str(X.shape[0])+"\n\n")


def testimage():
    text.delete('1.0', END)
    global X1
    global filename
    X2=[]
    #name = filedialog.askopenfilename(initialdir="testImages")
    filename = filedialog.askdirectory(initialdir = ".")
    folder = os.path.basename(filename)
    print(folder)
    for name in glob.glob(filename+"/*.*"):
        img = cv2.imread(name)
        img = cv2.resize(img, (150,100))
        XX = np.array(img)
        XX = XX.astype('int64')
        
        X2.append(XX.flatten())
        X1=np.array(X2)
    print(X1.shape)
    text.insert(END,'Test Images uploaded')

'''def realtime():
  global X1
  import cv2
  vid = cv2.VideoCapture(0)
  #vid.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
  #vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
  while(True):
      ret, frame = vid.read()
      cv2.imshow('frame', frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          img = cv2.imwrite("test1/NewPicture.jpg",frame)
          break
  vid.release()
  cv2.destroyAllWindows()
  img = cv2.resize(img, (150,100))
  XX = np.array(img)
  XX = XX.astype('int64')
  X1=(XX.flatten())'''

def values(val):
    if val == 0:
        msg = '10'
    elif val == 1:
        msg = '20'
    elif val == 2:
        msg = '50'
    elif val == 3:
        msg = '100'
    elif val == 4:
        msg = '200'
    elif val == 5:
        msg = '500'
    elif val == 6:
        msg = '2000'
    return msg
    
def SVC():
    text.delete('1.0', END)
    from sklearn.svm import SVC  
    classifier = SVC(kernel='poly')
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    value=classifier.predict(X1)
    text.insert(END,"Confusion matrix:\n")
    text.insert(END,confusion_matrix(y_test,y_pred))
    text.insert(END,"\nSVC - POLY\n")
    text.insert(END,(accuracy_score(y_test,y_pred)*100))
    text.insert(END,"\n")
    i=0
    for name in glob.glob(filename+"/*.*"):
        img = cv2.imread(name)
        img = cv2.resize(img,(450,450))
        print(value[i])
        textmsg=values(value[i])
        i=i+1
        cv2.putText(img, textmsg, (100, 200),  cv2.FONT_HERSHEY_SIMPLEX,3, (0, 0, 255), 5)
        cv2.imshow(textmsg,img)
        cv2.waitKey(0)
    

def DTC():
    text.delete('1.0', END)
    from sklearn.tree import DecisionTreeClassifier 
    algo1=DecisionTreeClassifier()
    algo1.fit(X_train,y_train)
    y_pred=algo1.predict(X_test)
    value=algo1.predict(X1)
    text.insert(END,"Confusion matrix:\n")
    text.insert(END,confusion_matrix(y_test,y_pred))
    text.insert(END,"\nDECISION TREE\n")
    text.insert(END,(accuracy_score(y_test,y_pred)*100))
    text.insert(END,"\n")
    i=0
    for name in glob.glob(filename+"/*.*"):
        img = cv2.imread(name)
        img = cv2.resize(img,(450,450))
        print(value[i])
        textmsg=values(value[i])
        i=i+1
        cv2.putText(img, textmsg, (100, 200),  cv2.FONT_HERSHEY_SIMPLEX,3, (0, 0, 255), 5)
        cv2.imshow(textmsg,img)
        cv2.waitKey(0)
    

def KNN():
    text.delete('1.0', END)
    from sklearn.neighbors import KNeighborsClassifier 
    from sklearn.model_selection import KFold,cross_val_score

    algo2=KNeighborsClassifier()
    algo2.fit(X_train,y_train)
    y_pred = algo2.predict(X_test)
    value=algo2.predict(X1)
    text.insert(END,"Confusion matrix:\n")
    text.insert(END,confusion_matrix(y_test,y_pred))
    text.insert(END,"\nKNN\n")
    text.insert(END,(accuracy_score(y_test,y_pred)*100))
    text.insert(END,"\n")
    i=0
    for name in glob.glob(filename+"/*.*"):
        img = cv2.imread(name)
        img = cv2.resize(img,(450,450))
        print(value[i])
        textmsg=values(value[i])
        i=i+1
        cv2.putText(img, textmsg, (100, 200),  cv2.FONT_HERSHEY_SIMPLEX,3, (0, 0, 255), 5)
        cv2.imshow(textmsg,img)
        cv2.waitKey(0)


def RFC():
    text.delete('1.0', END)
    from sklearn.ensemble import RandomForestClassifier  
    algo3=RandomForestClassifier(max_depth=5,random_state=10)
    algo3.fit(X_train,y_train)
    y_pred = algo3.predict(X_test)
    value=algo3.predict(X1)
    text.insert(END,"Confusion matrix:\n")
    text.insert(END,confusion_matrix(y_test,y_pred))
    text.insert(END,"\nRANDOM FOREST\n")
    text.insert(END,(accuracy_score(y_test,y_pred)*100))
    text.insert(END,"\n")
    i=0
    for name in glob.glob(filename+"/*.*"):
        img = cv2.imread(name)
        img = cv2.resize(img,(450,450))
        print(value[i])
        textmsg=values(value[i])
        i=i+1
        cv2.putText(img, textmsg, (100, 200),  cv2.FONT_HERSHEY_SIMPLEX,3, (0, 0, 255), 5)
        cv2.imshow(textmsg,img)
        cv2.waitKey(0)



font = ('times', 16, 'bold')
title = Label(main, text='Indian Currency Classification ', justify=LEFT)
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

predictButton = Button(main, text="Upload Test Images", command=testimage)
predictButton.place(x=560,y=100)
predictButton.config(font=font1)

font1 = ('times', 13, 'bold')
SVCButton = Button(main, text="Classification using SVC", command=SVC)
SVCButton.place(x=10,y=150)
SVCButton.config(font=font1)

KNNButton = Button(main, text="Classification using KNN", command=KNN)
KNNButton.place(x=280,y=150)
KNNButton.config(font=font1) 

DTCButton = Button(main, text="Classification using DTC", command=DTC)
DTCButton.place(x=560,y=150)
DTCButton.config(font=font1) 

RFCButton = Button(main, text="Classification using RFC", command=RFC)
RFCButton.place(x=10,y=200)
RFCButton.config(font=font1)

'''predictButton = Button(main, text="Capture Test Image", command=realtime)
predictButton.place(x=280,y=200)
predictButton.config(font=font1)'''

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1) 

main.config(bg='LightSteelBlue3')
main.mainloop()
