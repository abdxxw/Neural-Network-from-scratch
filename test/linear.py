import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from nn.Module import Linear
from nn.Loss import MSELoss
from tools import basic 


batchsize = 1000

datax, datay = basic.gen_arti(centerx=1, centery=1, sigma=0.4, nbex=batchsize, data_type=0, epsilon=0.1)
testx, testy = basic.gen_arti(centerx=1, centery=1, sigma=0.4, nbex=batchsize, data_type=0, epsilon=0.1)


datay = np.where(datay==-1,0,1).reshape((-1,1))
testy = np.where(testy==-1,0,1).reshape((-1,1))


n = datax.shape[1]
d = 1


iteration = 10
gradient_step = 1e-3

loss_mse = MSELoss()
lin_layer = Linear(n, d)

for _ in range(iteration):
    #forward
    hidden_l = lin_layer.forward(datax)
    
    #backward
    loss_back = loss_mse.backward(datay, hidden_l)
    delta_linear = lin_layer.backward_delta(datax, loss_back)

    lin_layer.backward_update_gradient(datax, loss_back)
    
    lin_layer.update_parameters(gradient_step=gradient_step)
    
    lin_layer.zero_grad()


def predict(x):
    hidden_l = lin_layer.forward(x)
    return np.where(hidden_l >= 0.5,1, 0)


print("accuracy : ",np.where(testy == predict(testx),1,0).mean())

basic.plot_frontiere(testx, predict, step=100)
basic.plot_data(testx, testy.reshape(-1))

