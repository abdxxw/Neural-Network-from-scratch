import numpy as np

class Sequentiel:
    
    def __init__(self, modules):
        assert(len(modules) > 1)
        self._modules = modules

    def forward(self, x):
        outputs = [x]
        for m in self._modules:
            outputs.append(m.forward(outputs[-1]))
        outputs.reverse()
        return outputs
        

    def backward(self, outputs, delta):

        list_delta = [delta]
        for i, module in enumerate(np.flip(self._modules)): # len(outputs) = len(modules) +1 so its safe
            module.backward_update_gradient(outputs[i+1], list_delta[-1])
            list_delta.append(module.backward_delta(outputs[i+1] , list_delta[-1]))
    
        return list_delta
    
    
    def update_parameters(self, eps = 1e-3):
        for m in self._modules:
            m.update_parameters(gradient_step=eps)
            m.zero_grad()
    
    def predict(self, x):
        return self.forward(x)[0]
    

class Optim:
    
    def __init__(self, net, loss, eps = 1e-3):
        self._net  = net
        self._loss = loss
        self._eps  = eps
        
    def step(self,batch_x, batch_y):
        
        outputs = self._net.forward(batch_x)
        loss = self._loss.forward(batch_y,outputs[0])
        delta = self._loss.backward(batch_y, outputs[0])
        self._net.backward(outputs, delta)
        self._net.update_parameters(self._eps)
        
        return loss
        
    def SGD(self, X, Y, batch_size, epoch=10):
        assert len(X) == len(Y)
        
        #shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        Y = Y[indices]
        
    
        #generate batch list
        batch_X  = [X[i:i + batch_size] for i in range(0, len(X), batch_size)]
        batch_Y = [Y[i:i + batch_size] for i in range(0, len(Y), batch_size)]
        mean = []
        std = []
        for e in range(epoch):
            tmp = []
            for x,y in zip(batch_X, batch_Y):
                tmp.append(self.step(x, y))
            tmp = np.asarray(tmp)
            mean.append(tmp.mean())
            std.append(tmp.std())
        return mean, std
                