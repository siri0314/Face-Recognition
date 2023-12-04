#solution for problem 2

import numpy as np
import glob
from scipy.misc import imread 

#train faces
train_faces=glob.glob('/Users/munni/Downloads/MIT-CBCL-Face-dataset/MIT-CBCL-Face-dataset/train/face/*.pgm')
listpgm=[]
for files in train_faces:
    image = imread(files)
    image = image.ravel()
    listpgm.append(image.ravel())
    
faces = range(len(train_faces)*361)
faces = (np.reshape(faces,(len(train_faces),361)))
for i in range(len(faces)):
    for j in range(361):
          faces[i][j] = listpgm[i][j]
#adding bias
faces=np.c_[faces,np.ones(faces.shape[0])]


#test faces
test_faces=glob.glob('/Users/munni/Downloads/MIT-CBCL-Face-dataset/MIT-CBCL-Face-dataset/test/face/*.pgm')
listpgm1=[]
for files1 in test_faces:
    image1 = imread(files1)
    image1=image1.ravel()
    listpgm1.append(image1.ravel())
    
faces1 = range(len(test_faces)*361)
faces1 = (np.reshape(faces1,(len(test_faces),361)))
for x in range(len(faces1)):
    for y in range(361):
          faces1[x][y] = listpgm1[x][y]
#adding bias
faces1=np.c_[faces1,np.ones(faces1.shape[0])]


#train non faces
train_non_faces=glob.glob('/Users/munni/Downloads/MIT-CBCL-Face-dataset/MIT-CBCL-Face-dataset/train/non-face/*.pgm')
listpgm2=[]
for files2 in train_non_faces:
    image2 = imread(files2)
    image2=image2.ravel()
    listpgm2.append(image2.ravel())
    
faces2 = range(len(train_non_faces)*361)
faces2 = (np.reshape(faces2,(len(train_non_faces),361)))
for a in range(len(faces2)):
    for b in range(361):
          faces2[a][b] = listpgm2[a][b]
#adding bias
faces2=np.c_[faces2,np.zeros(faces2.shape[0])]



#test non faces
test_non_faces=glob.glob('/Users/munni/Downloads/MIT-CBCL-Face-dataset/MIT-CBCL-Face-dataset/test/non-face/*.pgm')
listpgm3=[]
for files3 in test_non_faces:
    image3 = imread(files3)
    image3=image3.ravel()
    listpgm3.append(image3.ravel())
    
faces3 = range(len(test_non_faces)*361)
faces3 = (np.reshape(faces3,(len(test_non_faces),361)))
for p in range(len(faces3)):
    for q in range(361):
          faces3[p][q] = listpgm3[p][q]
#adding bias
faces3=np.c_[faces3,np.zeros(faces3.shape[0])]
train_df = np.concatenate((faces,faces2), axis=0)
test_df = np.concatenate((faces1,faces3), axis=0)

w=np.random.uniform(size=(train_df.shape[1],))
w=np.random.uniform(size=(test_df.shape[1],))

import pandas as pd
train_df=pd.DataFrame(train_df)
test_df=pd.DataFrame(test_df)
m=train_df.iloc[:,:-1].values
n=train_df.iloc[:,-1].values
m1=test_df.iloc[:,:-1].values
n1=test_df.iloc[:,-1].values
T=np.random.uniform(size=(m.shape[1],))

nEpoch=100
alpha=0.0111
lmda=0.9



def sigmoid(r):
  return 1 / (1 + np.exp(-r))
for epoch in np.arange(0,nEpoch):
    hypothesis=sigmoid(m.dot(T))
    error=hypothesis-n
    loss=np.sum(error**2)
    gradient=m.T.dot(error)-(lmda*T)
    T=T-alpha*gradient
    y_pred=sigmoid(m1.dot(T))
print("loss={:.7f}".format(loss))



from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
confmat=confusion_matrix(n1,y_pred)
acc=accuracy_score(n1, y_pred)
print(classification_report(n1, y_pred))

from sklearn import metrics

fpr,tpr,threshold = metrics.roc_curve(n1,y_pred)
roc_auc=metrics.auc(fpr,tpr)

import matplotlib.pyplot as plt
plt.plot(fpr,tpr,'b',label='AUC=%0.2f' %roc_auc)
plt.legend(loc= 'lower right')
plt.plot([0,1], [0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('tpr')
plt.xlabel('fpr')
plt.show()