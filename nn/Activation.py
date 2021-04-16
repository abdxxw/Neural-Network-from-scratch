import numpy as np
import nn.Module

class TanH(nn.Module):
    
    def forward(self, X):
        assert X.shape == (self.n, -1)
        
        return np.tanh(X)
    
    
    def backward_delta(self, input, delta):
        assert input.shape == (self.n, -1)
        assert delta.shape == (self.d, -1)

        return 1 - np.tanh(input)**2 * delta
    
class Sigmoid(nn.Module):
    
    def forward(self, X):
        assert X.shape == (self.n, -1)
        
        return 1 / (1 + np.exp(-X))
    
    
    def backward_delta(self, input, delta):
        assert input.shape == (self.n, -1)
        assert delta.shape == (self.d, -1)

        tmp = 1 / (1 + np.exp(-input))
        return delta * (tmp * (1-tmp))
    