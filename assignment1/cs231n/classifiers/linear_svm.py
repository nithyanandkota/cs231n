import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j]+=X[i]
        dW[:,y[i]]-=X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /=num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  dW += reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################





  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #print dW.shape
  num_classes = W.shape[1]
  num_train = X.shape[0]
  Pos_mat=np.zeros((num_train,num_classes))
  Pos_mat[np.arange(num_train),y]=1
  Neg_mat=np.ones((num_train,num_classes))
  Neg_mat[np.arange(num_train),y]=0
  scores=X.dot(W)
  #print np.shape(scores)
  #print np.shape(Pos_mat)
  scores_actual=(np.sum(scores*Pos_mat,axis=1)).reshape(num_train,1)
  #print scores_actual
  #print np.shape(scores_actual)
  marginall=(scores-scores_actual+1)*Neg_mat

  marginabove=np.array(marginall> 0, dtype=int)
  margin=marginall*marginabove
  #print np.shape(margin)
  dW=np.transpose(np.transpose(marginabove).dot(X)-np.transpose(Pos_mat*np.sum(marginabove,axis=1).reshape(num_train,1)).dot(X))
  loss=np.sum(margin)
  #print loss
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /=num_train
  dW += reg*W
  

  return loss, dW
