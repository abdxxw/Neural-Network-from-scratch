import numpy as np
from keras.losses import mse

from nn.Module import *
from scriptverif import maxpool1D

datax = np.random.randn(20, 10)
datay = np.random.choice([-1, 1], 20, replace=True)
linear = Linear(10, 1)
conv1D = Conv1D(k_size=3, chan_in=1, chan_out=5, stride=2)

linear.zero_grad()
conv1D.zero_grad()
res_conv = conv1D.forward(datax.reshape(20,10,1))
res_pool = maxpool1D.forward(res_conv)
res_flat = flatten.forward(res_pool)
res_linconv = linear.forward(res_flat)
res_mseconv = mse.forward(datay.reshape(-1,1),res_linconv)
delta_mseconv = mse.backward(datay.reshape(-1,1),res_linconv)
delta_linconv = linear.backward_delta(res_flat,delta_mseconv)
delta_flat = flatten.backward_delta(res_pool,delta_linconv)
delta_pool = maxpool1D.backward_delta(res_conv,delta_flat)
delta_conv= conv1D.backward_delta(datax.reshape(20,10,1),delta_pool)
conv1D.backward_update_gradient(datax.reshape(20,10,1),delta_pool)
grad_conv = conv1D._gradient
