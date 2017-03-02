import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  dW = np.zeros_like(W)
  for i in xrange(num_train):
    
    scores = X[i].dot(W)
    scores-= np.max(scores)
    correct_class_score = scores[y[i]]
    sumscores=np.sum(np.exp(scores))
    #if i==1:
       #print sumscores
       #print scores
       #print correct_class_score
       #print -(correct_class_score) + np.log(sumscores)
    loss += -(correct_class_score) + np.log(sumscores)
    
    for j in xrange(num_classes):
        if j == y[i]:
           dW[:,j]+=-X[i]+X[i]*np.exp(scores[j])/sumscores             
        else:                     
           dW[:,j]+=X[i]*np.exp(scores[j])/sumscores
 

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /=num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  dW += reg*W

  return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  num_classes = W.shape[1]
  num_train = X.shape[0]  
  loss = 0.0
  dW = np.zeros_like(W)
  Pos_mat=np.zeros((num_train,num_classes))
  Pos_mat[np.arange(num_train),y]=1

  scores = X.dot(W)
  max_scores=np.max(scores,axis=1)
  scores-= np.reshape(max_scores, (num_train,1))  
  
  sumscores= np.reshape(np.sum(np.exp(scores),axis=1),(num_train,1)) 
  scores_actual=(np.sum(scores*Pos_mat,axis=1)).reshape(num_train,1)
  #print sumscores
  #print scores[1]
  #print scores_actual[1]
  dW=np.transpose(X).dot(np.exp(scores)/sumscores)-np.transpose(X).dot(Pos_mat)
  #print np.exp(scores)[1]
  #print sumscores[1]
  #print (np.exp(scores)/sumscores) [1] 
  loss=  np.sum(-scores_actual +np.log(sumscores))
  
    
    
  loss /= num_train
  dW /=num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  dW += reg*W

  return loss, dW

