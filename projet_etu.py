import numpy as np


class Loss(object):
    
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass


class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        ## Annule gradient
        self._gradient = np.zeros((self.n,self.d))

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass

########################### Linéaire ##############################

class Lineaire(Module):
    
    def __init__(self, n, d, bias=True):
        self._n = n
        self._d = d
        self._bias = bias
        self._parameters = np.random.rand(n,d)
        self._gradient = np.zeros((n,d))
        if self._bias == True:
            self._bias_parameters = np.random.rand(1, d)
            self._bias_gradient = np.zeros((1, d))
        
    
    def forward(self, X):
        assert X.shape == (self.n, 1)
        
        return np.dot(X.T, self._parameters)

        
    def backward_update_gradient(self, input, delta):
        assert input.shape == (self.n, 1)
        assert delta.shape == (self.d, 1)

        self._gradient += np.dot(delta, input.T).T
    
    
    def backward_delta(self, input, delta):
        assert input.shape == (self.n, 1)
        assert delta.shape == (self.d, 1)
        
        out = np.dot(self._parameters.T, input)
        return out * delta



    
    
class MSELoss(Loss):
    
    def forward(self, y, yhat):
        return (y - yhat) ** 2
    
    def backward(self, y, yhat):
        return 2*(yhat - y) 
    


######################### Non-Linéaire ############################
        
    
class TanH(Module):
    
    def __init__(self, n, d):
        self.n = n
        self.d = d
        self._parameters = np.random.rand(n,d)
        self._gradient = np.zeros((n,d))
        
    
    def forward(self, X):
        assert X.shape == (self.n, 1)
        
        return np.tanh(X)

        
    def backward_update_gradient(self, input, delta):
        assert input.shape == (self.n, 1)
        assert delta.shape == (self.d, 1)

        self._gradient += np.dot(delta, input.T).T
    
    
    def backward_delta(self, input, delta):
        assert input.shape == (self.n, 1)
        assert delta.shape == (self.d, 1)
        
        out = 1 - np.tanh(input)**2
        return out * delta



class Sigmoide(Module):
    
    def __init__(self, n, d):
        self.n = n
        self.d = d
        self._gradient = np.zeros((n,d))
        
    def zero_grad(self):
        self._gradient = np.zeros((self.n,self.d))
    
    def forward(self, X):
        assert X.shape == (self.n, 1)
        
        return self.sigmoid(X)

        
    def backward_update_gradient(self, input, delta):
        assert input.shape == (self.n, 1)
        assert delta.shape == (self.d, 1)

        self._gradient += np.dot(delta, input.T).T
    
    
    def backward_delta(self, input, delta):
        assert input.shape == (self.n, 1)
        assert delta.shape == (self.d, 1)
        
        out = self.sigmoid_grad(input)
        return out * delta
    
    def sigmoid(self,x):                                     
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_grad(self,x):                                     
        return self.sigmoid(x) * (1 - self.sigmoid(x))


######################### Encapsulage ############################



class Sequentiel():
    
    def __init__(self, X, modules):
        assert(len(modules) > 1)
        self.n = modules
        self.X = X
        
    def forward_all(self):
        
        outputs = [self.X]
        data = self.X
        for m in self.modules:
            data = m.forward(data)
            outputs.append(data)
        
        return outputs     
    
    def backward_delta_all(self):
        
        outputs = self.forward_all()
        deltas = [outputs[-1]]
        for i in reversed(range(len(self.modules))):
            self.update_parameters()
            self.backward_update_gradient(outputs[i], deltas[0])
            delta = self.modules[i].backward_delta(outputs[i],deltas[0]) #i because len(outputs) = len(modules) + 1
            deltas = [delta] + deltas
        return deltas
    
    
    
######################### multi-classe ############################
        
class Mul