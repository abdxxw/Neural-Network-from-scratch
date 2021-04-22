
#import os, sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from nn.Module import Linear,TanH,Sigmoid
from nn.Loss import MSELoss
from tools import basic 



batchsize = 1000

datax, datay = basic.gen_arti(centerx=1, centery=1, sigma=0.4, nbex=batchsize, data_type=0, epsilon=0.1)
testx, testy = basic.gen_arti(centerx=1, centery=1, sigma=0.4, nbex=batchsize, data_type=0, epsilon=0.1)


datay = np.where(datay==-1,0,1).reshape((-1,1))
testy = np.where(testy==-1,0,1).reshape((-1,1))


n = datax.shape[1]
hidden = 5
d = 1


iteration = 10
gradient_step = 1e-5

loss_mse = MSELoss()
lin_layer = Linear(n, hidden)
lin_layer2 = Linear(hidden, d)
act_sig = Sigmoid()
act_tan = TanH()

for _ in range(iteration):
    
    #forward
    hidden_lin = lin_layer.forward(datax)    
    hidden_tan = act_tan.forward(hidden_lin)
    hidden_lin2 = lin_layer2.forward(hidden_tan)
    hidden_sig = act_sig.forward(hidden_lin2)
    loss = loss_mse.forward(datay,hidden_sig)
    
    #backward
    
    loss_back = loss_mse.backward(datay, hidden_sig)
    delta_sig = act_sig.backward_delta(hidden_lin2,loss_back)
    delta_lin2 = lin_layer2.backward_delta(hidden_tan,delta_sig)
    delta_tan = act_tan.backward_delta(hidden_lin,delta_lin2)
    delta_lin = lin_layer.backward_delta(datax,delta_tan)
    


    lin_layer2.backward_update_gradient(hidden_tan, delta_sig)
    lin_layer.backward_update_gradient(datax, delta_tan)    
    

    lin_layer2.update_parameters(gradient_step = gradient_step)
    lin_layer.update_parameters(gradient_step = gradient_step)
    
    lin_layer2.zero_grad()
    lin_layer.zero_grad()
    


def predict(x):
    hidden_l = lin_layer.forward(x)
    hidden_l = act_tan.forward(hidden_l)
    hidden_l = lin_layer2.forward(hidden_l)
    hidden_l = act_sig.forward(hidden_l)  
    return np.where(hidden_l >= 0.5,1, 0)

print("accuracy : ",np.where(testy == predict(testx),1,0).mean())

basic.plot_frontiere(testx, predict, step=100)
basic.plot_data(testx, testy.reshape(-1))

