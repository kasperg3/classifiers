import numpy as np
from random import shuffle
from past.builtins import xrange

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
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0


  for i in xrange(num_train):
    scores = X[i].dot(W)

    #For instabilities
    #scores = scores - np.max(scores)

    correct_class_score = scores[y[i]]

    #loss function
    denominator = np.sum(np.exp(scores))
    nominator = np.exp(correct_class_score)
    loss -= np.log(nominator/denominator)


    #calculate the gradients http://cs231n.github.io/neural-networks-case-study/#grad
    p_grad = np.exp(scores) / np.exp(scores).sum()
    p_grad[y[i]] -= 1

    #reshaping to be able to dot with each other
    #(cannot dot because of (3073,) has to be (3073,1))
    p_grad = p_grad.reshape(1,num_classes)
    X_t = X[i].reshape(X.shape[1],1)

    dW += np.dot(X_t, p_grad)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #Gradient
  dW /= num_train
  dW += 2*W*reg




  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)

  #loss function
  p_scores = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

  #calculates all yi loss scores
  p_scores_y = p_scores[np.arange(num_train), y]

  #sum of all loss' of yi
  loss = -1*np.sum(np.log(p_scores_y))

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)


  #Gradient calc
  p_scores[np.arange(num_train),y] -= 1

  dW = np.dot(np.transpose(X), p_scores)

  dW /= num_train
  dW += 2*W*reg


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

