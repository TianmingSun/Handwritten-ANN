import numpy as np
import MNISTtools

def normalize_MNIST_images(x):
    x = x.astype(np.float32)
    MAX = np.max(x)
    MIN = np.min(x)
    x = - 1 + 2 * ( x - MIN) / ( MAX - MIN)
    return x

def label2onehot(lbl):
    d = np.zeros((lbl.max() + 1, lbl.size))
    d[[ind for ind in lbl], np.arange(lbl.size)] = 1
    return d

def onehot2label(d):
    lbl = d.argmax(axis=0)
    return lbl

def relu(a):
    return np.maximum(a,0)
    
def relup(a, e):
    return (a > 0) * 1 * e

def init_shallow(Ni, Nh, No):
    b1 = np.random.randn(Nh, 1) / np.sqrt((Ni+1.)/2.)
    W1 = np.random.randn(Nh, Ni) / np.sqrt((Ni+1.)/2.)
    b2 = np.random.randn(No, 1) / np.sqrt((Nh+1.))
    W2 = np.random.randn(No, Nh) / np.sqrt((Nh+1.))
    return W1, b1, W2, b2

def forwardprop_shallow(x, net):
    W1 = net[0]
    b1 = net[1]
    W2 = net[2]
    b2 = net[3]
    a1 = W1 @ x + b1
    h1 = relu(a1)
    a2 = W2 @ h1 + b2
    y = softmax(a2)
    return y

def eval_loss(y, d):
    return np.sum( -d * np.log(y)) / y.size

def eval_perfs(y, lbl):
    dlbl = label2onehot(lbl)
    pred = np.max( y * dlbl, axis=0)
    comp = np.max( y, axis=0)
    return np.sum(pred != comp) / np.size(lbl)

def update_shallow(x, d, net, gamma=.05):
    W1 = net[0]
    b1 = net[1]
    W2 = net[2]
    b2 = net[3]
    Ni = W1.shape[1]
    Nh = W1.shape[0]
    No = W2.shape[0]
    gamma = gamma / x.shape[1] # normalized by the training dataset size
    # Forward phase
    a1 = W1 @ x + b1
    h1 = relu(a1)
    a2 = W2 @ h1 + b2
    y = softmax(a2)
    # Error evaluation
    e = -d / y
    # Backward phase
    delta2 = softmaxp(a2, e)
    delta1 = relup(a1, W2.T @ delta2)
    # Gradient update
    W2 = W2 - gamma * delta2 @ h1.T
    W1 = W1 - gamma * delta1 @ x.T
    b2 = b2 - gamma * delta2.sum( axis=1, keepdims=True)
    b1 = b1 - gamma * delta1.sum( axis=1, keepdims=True)
    return W1, b1, W2, b2

def backprop_shallow(x, d, net, T=5, B=100, gamma=.05):
    N = x.shape[1]
    NB = int((N+B-1)/B) # ceil(N/B)
    lbl = onehot2label(d)
    loss = []
    perfs = []
    for t in range(T):
        shuffled_indices = np.random.permutation(range(N))
        for l in range(NB):
            minibatch_indices = shuffled_indices[B*l:min(B*(l+1), N)]
            mini = x[:,minibatch_indices]
            net = update_shallow(mini, d[:,minibatch_indices], net, gamma)
        y = forwardprop_shallow(x, net)
        loss.append(eval_loss(y,d))
        perfs.append(eval_perfs(y,lbl))     
    print ("Final Loss:", loss[-1])
    print ("Final Percentage of Training Errors: {:.2f} %".format(perfs[-1] * 100))

xtrain, ltrain = MNISTtools.load(dataset="training")
xtrain = normalize_MNIST_images(xtrain)
dtrain = label2onehot(ltrain)
xtest, ltest = MNISTtools.load(dataset="testing")
xtest = normalize_MNIST_images(xtest)

net = backprop_shallow(xtrain, dtrain, netinit, T=5, B=100)
pred = forwardprop_shallow(xtest, net)
print('The size of testing set is {}.'.format(np.shape(xtest)[1]))
print('The error rate of the network on the test set is {:.2f} %.'.format(eval_perfs(pred, ltest) * 100))