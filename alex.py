import numpy as np
import os
import cv2

def uploaddata(path):
    global folder
    folder=os.path.basename(path)
    print(folder)

def preprocessing():
    label = ['Fake','Real']

    X = []
    Y = []

    for i in range(len(label)):
        for root, dirs, directory in os.walk(folder+'/'+label[i]):
            for j in range(len(directory)):
                img = cv2.imread(folder+'/'+label[i]+"/"+directory[j])
                img = cv2.resize(img,(227,227))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(227,227,3)
                XX = np.asarray(im2arr)
                XX = XX.astype('float32')
                XX = XX/255
                X.append(XX)
                Y.append(i)
                print(folder+'/'+label[i]+"/"+directory[j]+" "+str(X[j].shape))
                        
    np.save("model/features.txt",X)
    np.save("model/labels.txt",Y)

    X = np.load("model/features.txt.npy")
    Y = np.load("model/labels.txt.npy")


    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
