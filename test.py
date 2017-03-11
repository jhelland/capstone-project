
# computing & data structures
import numpy as np
import theano
import theano.tensor as T

# graphics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn

# user files
from logistic_regression import *


def run(n_iterations=100, n_obs=1000):

    # generate some test data
    print('...Generating cluster data')
    N = n_obs
    labels = np.random.random_integers(0, 1, N)
    means = np.array([[0, 0, 0], [-1, 1, 1]])
    temp_matrix = np.array([np.random.random_sample((3, 3)),
                            np.random.random_sample((3, 3))])
    covariances = np.array([np.dot(temp_matrix[0].T, temp_matrix[0]),
                            np.dot(temp_matrix[1].T, temp_matrix[1])])
    data = np.vstack([np.random.multivariate_normal(means[i], covariances[i])
                      for i in labels])
    data = data[:, 0:2]  # remove 3rd dimension because I'm lazy

    # cast data as proper dtype
    data = data.astype(theano.config.floatX)
    labels = labels.astype('int32')

    # symbolic variables and functions
    print('...Constructing model')
    data_sym = T.matrix('data')
    labels_sym = T.ivector('labels')

    # instantiate logistic regression object
    classifier = LogisticRegression(input=data_sym,
                                    n_in=2,
                                    n_out=2)

    # instantiate symbolic cost function
    cost = (
        classifier.negative_log_likelihood(labels_sym)
        # regularization terms can go here
    )

    # compile callable theano function to compute model prediction
    predict = theano.function(inputs=[classifier.input],
                              outputs=classifier.y_pred)

    # compute symbolic gradients
    grads = [T.grad(cost=cost, wrt=param) for param in classifier.params]

    # parameter training update rules
    learning_rate = 0.01
    updates = [(param, param - learning_rate * grad)
               for param, grad in zip(classifier.params, grads)]

    # model training function
    train = theano.function(
        inputs=[],
        outputs=cost,
        updates=updates,
        givens={
            data_sym: data,
            labels_sym: labels
        }
    )

    # train the model with standard batch gradient descent
    print('...Training model')
    for i in range(n_iterations):
        current_cost = train()
        print('\t iteration %i/%i \t cost = %f' % (i, n_iterations, current_cost))

    # calculate prediction for data from trained model
    prediction = predict(data)

    # plot classification from model alongside original data
    print('...Plotting')
    fig = plt.figure(figsize=plt.figaspect(0.5))
    plt.axis('off')
    ax = fig.add_subplot(121)
    ax.scatter(data[:, 0], data[:, 1],
               c=labels, cmap=cm.autumn)

    ax = fig.add_subplot(122)
    ax.scatter(data[:, 0], data[:, 1],
               c=prediction, cmap=cm.autumn)

    plt.show()
    plt.clf()


if __name__ == '__main__':
    run(n_iterations=100, n_obs=1000)