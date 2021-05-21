import numpy as np

class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        ## Annule gradient
        pass

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
    
    

class Linear(Module):
    
    def __init__(self, n, d, bias=True,type=2):
        self._n = n
        self._d = d
        self._gradient = np.zeros((n,d))
        if type == 0:
            self._parameters = np.random.random((n,d)) - 0.5
        if type == 1:
            self._parameters = np.random.normal(0, 1,(n,d))
        if type == 2:
            self._parameters = np.random.normal(0, 1,(n,d))*np.sqrt(2/(n+d))
            
        if bias == True:
            if type == 0:
                self._bias = np.random.random((1,d)) - 0.5
            if type == 1:
                self._bias = np.random.normal(0, 1,(1,d))
            if type == 2:
                self._bias = np.random.normal(0, 1,(1,d))*np.sqrt(2/(n+d))
            self._bias_grad = np.zeros((1, d))
        else:
            self._bias = None
        
    
    def forward(self, X):
        assert X.shape[1] == self._n  # batch * input
        
        if self._bias is not None:
            return np.dot(X, self._parameters) + self._bias #batch * output
        else:
            return np.dot(X, self._parameters) #batch * output

            
    def backward_update_gradient(self, input, delta):
        assert input.shape[1] == self._n
        assert delta.shape[1] == self._d
        assert input.shape[0] == delta.shape[0]

        self._gradient += np.dot(input.T, delta)
        
        if self._bias is not None:
            self._bias_grad += np.sum(delta, axis = 0)
    
    
    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= gradient_step*self._gradient
        
        if self._bias is not None:
            self._bias -= gradient_step*self._bias_grad
    
    def backward_delta(self, input, delta):
        assert input.shape[1] == self._n
        assert delta.shape[1] == self._d
      

        return np.dot(delta,self._parameters.T) 



    def zero_grad(self):
        self._gradient = np.zeros((self._n,self._d))
        self._bias_grad = np.zeros((1, self._d))


class TanH(Module):
    
    def forward(self, X):
        return np.tanh(X)
    
    
    def backward_delta(self, input, delta):
        return (1 - np.tanh(input)**2) * delta
    
    def update_parameters(self, gradient_step=1e-3):
        pass
    
    
class Sigmoid(Module):
    
    def forward(self, X):
        return 1 / (1 + np.exp(-X))
    
    
    def backward_delta(self, input, delta):

        tmp = 1 / (1 + np.exp(-input))
        return delta * (tmp * (1-tmp))
    
    def update_parameters(self, gradient_step=1e-3):
        pass
    

class Softmax(Module):
    
    def forward(self, X):
        expo = np.exp(X)
        return expo / np.sum(expo, axis=1).reshape((-1,1))

    def backward_delta(self, input, delta):
        
        expo = np.exp(input)
        out = expo / np.sum(expo, axis=1).reshape((-1,1))
        return delta * (out * (1-out))
    
    def update_parameters(self, gradient_step=1e-3):
        pass
    
    