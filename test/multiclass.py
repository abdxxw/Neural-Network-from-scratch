
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from nn.Module import Linear,TanH,Sigmoid,Softmax
from nn.Loss import CElogSoftMax,MSELoss,CELoss,BCELoss
from tools.nntools import Sequentiel, Optim
from tools.basic import load_usps, show_image,draw_pred
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.datasets import mnist

def onehot(x): # single digit
    out = [0] * 10
    out[x] = 1
    return out



    
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# building the input vector from the 28x28 pixels
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255


X_train = X_train[:5000]
y_train = y_train[:5000]
X_test = X_train[:1000]
y_test = y_train[:1000]



y_train = np.asarray([onehot(x) for x in y_train])

type=2

n = X_train.shape[1]
hidden1 = 128
hidden2 = 64
d = 10


iteration = 100
gradient_step = 1e-4
batchsize = 100

def label_func(x):
    return np.argmax(x,axis=1)

lin_layer = Linear(n, hidden1,type=type)
lin_layer2 = Linear(hidden1, hidden2,type=type)
lin_layer3 = Linear(hidden2, d,type=type)
act_softmax = Softmax()
act_tan = TanH()
act_tan2 = TanH()
loss = CElogSoftMax()



net = Sequentiel([lin_layer,act_tan,lin_layer2,act_tan,lin_layer3],labels=label_func)

opt = Optim(net,loss,eps=gradient_step)
mean, std = opt.SGD(X_train,y_train,batchsize,iteration,earlystop=50)
plt.figure()
plt.plot(mean)
plt.plot(std)
plt.legend(('Mean', 'std'))
plt.show()



print("accuracy : ",opt.score(X_test,y_test))


draw_pred(X_test,y_test,opt._net,6,n=28)