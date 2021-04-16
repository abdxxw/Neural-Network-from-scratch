
#import os, sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from nn.Module import Linear,TanH,Sigmoid
from nn.Loss import MSELoss
from tools import basic
from tools.nntools import Sequentiel, Optim
import matplotlib.pyplot as plt



batchsize = 1000

datax, datay = basic.gen_arti(centerx=1, centery=1, sigma=0.1, nbex=batchsize, data_type=0, epsilon=0.1)
testx, testy = basic.gen_arti(centerx=1, centery=1, sigma=0.1, nbex=batchsize, data_type=0, epsilon=0.1)


datay = np.where(datay==-1,0,1).reshape((-1,1))
testy = np.where(testy==-1,0,1).reshape((-1,1))


n = datax.shape[1]
hidden = 5
d = datay.shape[1]


iteration = 10
gradient_step = 1e-5

loss_mse = MSELoss()
lin_layer = Linear(n, hidden)
lin_layer2 = Linear(hidden, d)
act_sig = Sigmoid()
act_tan = TanH()

net = Sequentiel([lin_layer,act_tan,lin_layer2,act_sig])


opt = Optim(net,loss_mse,eps=gradient_step)
mean, std = opt.SGD(datax,datay,batchsize,iteration)
print(mean)
plt.figure()
plt.plot(mean)
plt.plot(std)
plt.legend(('Mean', 'std'))
plt.show()

def predict(x):
    return np.where(net.predict(x) >= 0.5,1, -1)

basic.plot_frontiere(testx, predict, step=100)
basic.plot_data(testx, testy.reshape(-1))

