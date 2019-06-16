import numpy as np
from matplotlib import pyplot
import MNISTtools

def normalize_MNIST_images(x):
    x = x.astype(np.float32)
    MAX = np.max(x)
    MIN = np.min(x)
    x = - 1 + 2 * ( x - MIN) / ( MAX - MIN)
    return x

def onehot2label(d):
    lbl = d.argmax(axis=0)
    return lbl

def label2onehot(lbl):
    d = np.zeros((lbl.max() + 1, lbl.size))
    d[[ind for ind in lbl], np.arange(lbl.size)] = 1
    return d

def softmax(a):
    d = np.exp(a - a.max(axis=0))
    return d / np.sum(d,axis=0)

def softmaxp(a, e):
    g = softmax(a)
    return g * e - np.sum(g * e, axis=0) * g

def relu(a):
    return np.maximum(a,0)
    
def relup(a, e):
    return (a > 0) * 1 * e

def eval_loss(y, d):
    return np.sum( -d * np.log(y)) / y.size

def eval_perfs(y, lbl):
    dlbl = label2onehot(lbl)
    pred = np.max( y * dlbl, axis=0)
    comp = np.max( y, axis=0)
    return np.sum(pred != comp) / np.size(lbl)

def init_fc(dim_in, dim_out, internal=[512]):
    layer = len(internal)+1
    internal.insert(0, dim_in)
    internal.append(dim_out)
    W = []
    b = []
    for i in range(layer):
        W_add = np.random.randn(internal[i+1], internal[i]) / np.sqrt((internal[i]+1.)/2.)
        b_add = np.random.randn(internal[i+1], 1) / np.sqrt((internal[i]+1.)/2.)
        W.append(W_add)
        b.append(b_add)
    return W,b

def forward(x, W, b):
    layer = len(W)
    h = x
    cache = []
    for i in range(layer):
        a = W[i] @ h + b[i]
        cache.append(a)
        h = relu(a)
    y = softmax(cache[-1])
    return y,cache

def backward(x, d, W, b, gamma=.05):
    layer = len(W)
    gamma = gamma / x.shape[1] # normalized by the training dataset size
    y,cache = forward(x, W, b)
    # Error evaluation
    e = -d / y
    # Backward phase
    delta = [softmaxp(cache[-1], e)]
    for i in range(layer-1,0,-1):
        add = relup(cache[i-1], W[i].T @ delta[-1])
        delta.append(add)
    delta = delta[::-1]
    # Gradient update
    W[0] = W[0] - gamma * delta[0] @ x.T
    b[0] = b[0] - gamma * delta[0].sum( axis=1, keepdims=True)
    for i in range(1,layer):
        W[i] = W[i] - gamma * delta[i] @ cache[i-1].T
        b[i] = b[i] - gamma * delta[i].sum( axis=1, keepdims=True)
    return W,b

def backprop_batch(x, d, v, d_v, W, b, T=10, B=100, gamma=.05):
    N = x.shape[1]
    NB = int((N+B-1)/B)
    lbl = onehot2label(d)
    lbl_v = onehot2label(d_v)
    loss = []
    perfs = []
    val = []
    for t in range(T):
        shuffled_indices = np.random.permutation(range(N))
        for k in range(NB):
            batch_indices = shuffled_indices[B*k:min(B*(k+1), N)]
            W,b = backward(x[:,batch_indices], d[:,batch_indices], W, b, gamma)
        y,_ = forward(x, W, b)
        y_v,_ = forward(v, W, b)
        loss.append(eval_loss(y,d))
        perfs.append(eval_perfs(y,lbl))
        val.append(eval_perfs(y_v,lbl_v))
        print(f'Process: {100*(t+1)/T:.2f}%')
        print(f"Loss: {loss[-1]}")
        print(f'Percentage of Training Errors: {100*perfs[-1]:.2f} %')
        print('======================================================')
    print('Done!')    
    print("Final Loss:", loss[-1])
    print(f'Final Percentage of Training Errors: {100*perfs[-1]:.2f} %')
    return W,b