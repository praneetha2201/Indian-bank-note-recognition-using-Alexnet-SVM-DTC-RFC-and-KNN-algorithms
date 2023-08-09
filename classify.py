import numpy as np
import os
import cv2

def uploaddata(path):
    global folder
    folder=os.path.basename(path)
    print(folder)

def preprocessing():
    label=['10','20','50','100','200','500','2000']
    X=[]
    Y=[]
    for i in range(len(label)):
        for root, dirs, directory in os.walk(folder+'/'+label[i]):
            for j in range(len(directory)):
                img = cv2.imread(folder+'/'+label[i]+'/'+directory[j])
                img = cv2.resize(img, (150,100))
                XX = np.array(img)
                XX = XX.astype('int64')
                X.append(XX.flatten())
                Y.append(i)
                print('dataset/'+label[i]+"/"+directory[j]+" "+str(X[j].shape))
                    
    np.save("model/features.txt",X)
    np.save("model/labels.txt",Y)

    X = np.load("model/features.txt.npy")
    Y = np.load("model/labels.txt.npy")

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    
    print(X.shape)
    print(Y.shape)
