# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
np.random.seed(0)
from scipy.special import logsumexp
import numpy.linalg
# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    x = x_train
    y = y_train
    test_datum = test_datum.reshape(test_datum.shape[0], 1)
    n = np.exp(-l2(x, test_datum.T) / (2*tau**2))
    sup = np.amax(n)
    for i in range(len(n)):
        n[i] -= sup
    m = np.exp(logsumexp(n))
    ai = n / m
    A = np.eye(len(x))
    for j in range(len(ai)):
        A[j][j] = ai[j]
        
    w = np.linalg.solve(np.dot(x.T, np.dot(A,x)) + 
                         lam*np.eye(x.shape[1]), np.dot(np.dot(x.T, A), y))
    return np.dot(test_datum.T, w)




def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    l = len(taus)
    l *= val_frac
    l = int(l)
    avg_val_loss = []
    avg_train_loss = []
    val_index = list(np.random.choice(range(len(x)), l, replace=False))
    for i in range(len(taus)):
        tau = taus[i]
        batch_train_loss = []
        batch_val_loss = []
        for j in range(len(x)):
            test_datum = x[j]
            y_hat = LRLS(test_datum, x, y, tau)
            loss = 0.5*(y[j] - y_hat)**2
            if j in val_index:
                batch_val_loss.append(loss)
            else:
                batch_train_loss.append(loss)

        avg_val_loss.append(sum(batch_val_loss)/len(batch_val_loss))
        avg_train_loss.append(sum(batch_train_loss)/len(batch_train_loss))
            
    return avg_train_loss, avg_val_loss


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    train_losses, test_losses = run_validation(x,y,taus,val_frac=0.3)
    plt.semilogx(train_losses)
    plt.semilogx(test_losses)

