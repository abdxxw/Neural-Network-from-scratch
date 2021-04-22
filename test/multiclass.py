
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from nn.Module import Linear,TanH,Sigmoid,Softmax
from nn.Loss import CElogSoftMax,MSELoss
from tools.nntools import Sequentiel, Optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def onehot(x): # single digit
    out = [0] * 10
    out[x] = 1
    return out

mnist = load_digits()

datax, datay = (mnist.data, mnist.target)


X_train, X_test, y_train, y_test = train_test_split(datax, datay, test_size=0.3)

y_train = np.asarray([onehot(x) for x in y_train])


n = datax.shape[1]
hidden1 = 30
hidden2 = 15
d = 10


iteration = 10
gradient_step = 1e-4
batchsize = 200

def label_func(x):
    return np.argmax(x,axis=1)

lin_layer = Linear(n, hidden1)
lin_layer2 = Linear(hidden1, hidden2)
lin_layer3 = Linear(hidden2, d)
act_softmax = Softmax()
act_tan = TanH()
act_tan2 = TanH()
loss = CElogSoftMax()

net = Sequentiel([lin_layer,act_tan,lin_layer2,act_tan2,lin_layer3,act_softmax],labels=label_func)

opt = Optim(net,loss,eps=gradient_step)
mean, std = opt.SGD(X_train,y_train,batchsize,iteration)
plt.figure()
plt.plot(mean)
plt.plot(std)
plt.legend(('Mean', 'std'))
plt.show()



print("accuracy : ",opt.score(X_test,y_test))
