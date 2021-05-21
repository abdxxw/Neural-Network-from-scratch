import numpy as np
import copy 

class Sequentiel:
    
    def __init__(self, modules,labels=None):
        assert(len(modules) > 0)
        self._modules = modules
        self._labels = labels

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
        if self._labels is not None:
            return self._labels(self.forward(x)[0])
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
        
    def SGD(self, X, Y, batch_size, epoch=10,earlystop=100):
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
        minloss=float("inf")
        bestepoch = 0
        stop=0
        bestModel = self._net
        for e in range(epoch):
            tmp = []
            for x,y in zip(batch_X, batch_Y):
                tmp.append(np.asarray(self.step(x, y)).mean())
            tmp = np.asarray(tmp)
            loss = tmp.mean()
            stop+=1
            if(loss < minloss):
                stop=0
                bestepoch = e
                minloss = loss
                bestModel = copy.deepcopy(self._net)
            if stop == earlystop:
                print("early stop best epoch : ",bestepoch)
                break
            mean.append(loss)
            std.append(tmp.std())
        self._net = bestModel
        return mean, std
                
    def score(self,x,y):
        return np.where(y == self._net.predict(x),1,0).mean()