import random

class Module:

    def zero_grad(self,):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self,):
        return []
    
class Neuron(Module):
    
    def __init__(self, nin: int, nonlinear=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlinear = nonlinear
    
    def __call__(self, x):
        act = sum((w*x for w, x in zip(self.w, x)), self.b)
        return act.relu() if self.nonlinear else act
    
    def parameters(self,):
        return self.w + [self.b]
    
    def __repr__(self,):
        return f"{'ReLU' if self.nonlinear else 'Linear'} Neuron({len(self.w)})"

class Layer(Neuron):
    
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
    
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self,):
        return [n.parameters() for n in self.neurons]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Layer):
    
    def __init__(self, nin, layers):
        sizes = [nin] + layers
        self.layers = [Layer(sizes[i], sizes[i+1], nonlinear=i!=len(sizes)-1) for i in range(len(sizes)-1)]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self,):
        return [p.parameters() for p in self.layers]