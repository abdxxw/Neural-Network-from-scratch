#import os, sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from nn.Module import Linear
from nn.Loss import MSELoss
from tools import basic 


batchsize = 1000

datax, datay = basic.gen_arti(centerx=1, centery=1, sigma=0.1, nbex=batchsize, data_type=0, epsilon=0.1)
testx, testy = basic.gen_arti(centerx=1, centery=1, sigma=0.1, nbex=batchsize, data_type=0, epsilon=0.1)


datay = np.where(datay==-1,0,1).reshape((-1,1))
testy = np.where(testy==-1,0,1).reshape((-1,1))


n = datax.shape[1]
d = datay.shape[1]


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
        return np.where(hidden_l >= 0.5,1, -1)
    
    basic.plot_frontiere(testx, predict, step=100)
    basic.plot_data(testx, testy.reshape(-1))


alpha = 562
beta = 29
bias = 10

def f(x1, x2):
    return x1 * alpha - x2 * beta + bias

def noise(x1, x2):
    bruit = np.random.normal(0, 1, len(x1)).reshape((-1, 1))
    return f(x1, x2) + bruit



batchsize = 100
x1 = np.random.uniform(-5, 5, batchsize).reshape((-1, 1))
x2 = np.random.uniform(-5, 5, batchsize).reshape((-1, 1))

datay = noise(x1, x2)
datax = np.concatenate((x1, x2), axis=1)


n = datax.shape[1]
d = datay.shape[1]


iteration = 100
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

x1 = np.random.uniform(-5, 5, batchsize).reshape((-1, 1))
x2 = np.random.uniform(-5, 5, batchsize).reshape((-1, 1))

testy = f(x1, x2)
testx = np.concatenate((x1, x2), axis=1)

hidden_l = lin_layer.forward(testx)
print("parameters :", str(lin_layer._parameters))
print("true:", str(alpha),str(beta))
print("bias:", str(lin_layer._bias))
print("true:", str(bias))



