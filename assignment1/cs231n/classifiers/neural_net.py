import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    # Compute the forward pass

    # (N, D) x (D, H) = (N, H) => N samples, each encoded into a vector of H
    # "hidden" features
    # Put simpler, imagine you only have 1 sample, a picture of D pixels.
    # Then the math would be (1, D) x (D, H) = (1, H)
    # Your 1 sample has D pixels. The (D, H) matrix is a a collection of H column vectors.
    # Each column vector is a "template". When multiplied (dot product) by that template,
    # you're projecting the picture onto that "template", and you get a "similarity" score.
    # How do you convert an image of D pixels into a single score? Easy,
    # (1, D) x (D, 1) => [ pixel_1 pixel_2 ... pixel_D ] * [ pixel_1_weight pixel_2_weight ... pixel_D_weight ]
    # Great, now you have a "similarity" score for the first template.
    # Because we're converting an image into H scores, however, we need H templates.
    # In other words, we need a resulting matrix (1, D) x (D, H) = (1, H).
    # This (1, H) matrix tells us all H templates' scores.
    # Finally, we're not computing just 1 image's scores. We're computing N images.
    # Thus, we get an (N, H) matrix.

    # notice X * W1 = (N, D) x (D, H) = (N, H) matrix
    # and yet we're adding a (H,) vector. With broadcasting, we actually
    # add this (H,) vector to every single row.

    # ReLU = max(0, W1 * X + b), where max is an element-wise operation
    hidden_layer = np.maximum(0, np.dot(X, W1) + b1)

    # scores = (N, H) * (H, C) = (N, C)
    # Each one of the N samples has C "scores", one score for each of the C classes.
    # For example, each of the N pictures has C scores, a dog score, a cat score, etc.
    scores = np.dot(hidden_layer, W2) + b2
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    # The loss will tell us the difference between the real class score and the
    # computed scores from above. We want to minimize the difference.
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################

    # The following is a vectorized implementation of softmax.
    # First, np.exp(scores) takes every element in the N x C matrix, and exponentiates it.
    # Next, np.sum does multiple things.
    # exps is a 2D matrix. Axis 0 is vertically down. Axis 1 is vertically across.
    # (See https://docs.scipy.org/doc/numpy-1.10.0/glossary.html#term-along-an-axis for a good explanation of axis).
    # We want an N x 1 matrix, each row containing the sum of the row's exps. Hence, we set axis=1
    # to sum each individual row up.
    # Finally, keepdims=True ensures that the sum of each row remains in its own row. This means
    # we'll end up with a N x 1 vector, instead of a 1 x N vector. If you're curious,
    # print np.sum(exps, axis=1) and print np.sum(exps, axis=1, keepdims=True)
    # to see the difference.
    exps = np.exp(scores)
    prob = exps / np.sum(exps, axis=1, keepdims=True)

    # Here, we compute the softmax L_i, the loss for each training sample.
    # Refer to http://cs231n.github.io/linear-classify/#softmax-classifier for the equation.
    # f_(y_i) is interpreted as follows:
    # First, get y_i for each training sample i.
    # For example, training sample 1 has y_1 = 2. This means training sample 1 is class 2.
    # That means you need the 2nd element from class scores f.
    # Because we've already computed the probabilities of each class, we can just choose the
    # 2nd element from the list of probabilities.
    # The following code pulls all rows from prob, and from each row, pull the y_i-th element.
    # See the slicing example from http://cs231n.github.io/python-numpy-tutorial/#numpy-array-indexing.
    losses = - np.log(prob[range(N), y])
    data_loss = np.sum(losses) / N
    regularization_loss_1 = np.sum(np.square(W1)) # also np.sum(W1*W1)
    regularization_loss_2 = np.sum(np.square(W2)) # also np.sum(W2*W2)
    loss = data_loss + 0.5 * reg * (regularization_loss_1 + regularization_loss_2)
    grads = {}
    return loss, grads

    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    pass
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


