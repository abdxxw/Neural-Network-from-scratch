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
    
    

class Lineaire(Module):
    
    def __init__(self, n, d, bias=True):
        self._n = n
        self._d = d
        self._parameters = np.random.rand(n,d)
        self._gradient = np.zeros((n,d))
        if bias == True:
            self._bias = np.random.rand(d, 1)
            self._bias_grad = np.zeros((d, 1))
        else:
            self._bias = None
        
    
    def forward(self, X):
        assert X.shape == (self._n, -1)  # input * batch
        
        if self._bias is not None:
            return np.dot(X.T, self._parameters).T + self._bias #output * output
        else:
            return np.dot(X.T, self._parameters).T #output * output

            
    def backward_update_gradient(self, input, delta):
        assert input.shape == (self.n, -1)
        assert delta.shape == (self.d,-1)
        assert input.shape[1] == delta.shape[1]

        self._gradient += np.dot(input, delta.T)
        
        if self._bias is not None:
            self._bias_grad += np.sum(delta, axis = 1)
    
    
    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= gradient_step*self._gradient
        
        if self._bias is not None:
            self._bias -= gradient_step*self._bias_grad
    
    def backward_delta(self, input, delta):
        assert input.shape == (self.n, -1)
        assert delta.shape == (self.d, -1)
        
        out = np.dot(self._parameters.T, input)
        return out * delta



    
    