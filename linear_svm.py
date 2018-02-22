import numpy as np
from random import shuffle
from past.builtins import xrange

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
    number_of_incorrect_classes = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        number_of_incorrect_classes += 1
        loss += margin
        dW[:,j] += X[i]
    dW[:,y[i]] -= number_of_incorrect_classes * X[i]



  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)


  #make dW an average
  dW /= num_train

  # Add regularication to the loss
  dW += reg * np.sum(W * W)

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

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  #num_classes = W.shape[1]
  num_train = X.shape[0]
  delta = 1


  scores = X.dot(W)
  correct_scores = np.matrix(scores[np.arange(scores.shape[0]),y])
  trans_correct = np.transpose(correct_scores)
  margin = scores - trans_correct + delta
  max_margin = np.maximum(0,margin)
  max_margin[np.arange(num_train),y] = 0
  loss = (np.sum(max_margin))/num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)



  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #counter = max_margin
  #counter[max_margin > 0] = 1

  counter = max_margin
  counter = np.where(max_margin > 0, 1, counter)

  number_of_counts = np.sum(counter,axis=1)
  #Reshaping (500,1) to (500,), making the multiplication possible
  number_of_counts = number_of_counts.reshape(num_train,)
  counter[np.arange(num_train),y] = -number_of_counts
  X_trans = np.transpose(X)
  dW = X_trans.dot(counter)

  dW /= num_train
  dW += reg * np.sum(W * W)

  return loss, dW