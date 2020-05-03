import sklearn
import numpy as np

def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array.

    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    return np.sqrt(np.dot(x, x.T))

def asymptotic_decay(learning_rate, t, max_iter):
    """Decay function of the learning process.
    Parameters
    ----------
    learning_rate : float
        current learning rate.

    t : int
        current iteration.

    max_iter : int
        maximum number of iterations for the training.
    """
    return learning_rate / (1+t/(max_iter/2))

def _build_iteration_indexes(data_len, num_iterations,
                             verbose=False, random_generator=None):
    """Returns an iterable with the indexes of the samples
    to pick at each iteration of the training.

    If random_generator is not None, it must be an instalce
    of numpy.random.RandomState and it will be used
    to randomize the order of the samples."""
    iterations = np.arange(num_iterations) % data_len
    if random_generator:
        random_generator.shuffle(iterations)
    
    return iterations

class VQTAM(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin):
    def __init__(self,
                grid_x_size=10,
                grid_y_size=10,
                output_length=0,
                sigma=1.0,
                learning_rate=0.5,
                iteration_count=500,
                decay_function=asymptotic_decay,
                neighborhood_function='gaussian',
                topology='rectangular',
                activation_distance='euclidean',
                random_seed=None):

        if sigma >= grid_x_size or sigma >= grid_y_size:
            warn('Warning: sigma is too high for the dimension of the map.')

        self._random_generator = np.random.RandomState(random_seed)

        self._learning_rate = learning_rate
        self._sigma = sigma
        self._iteration_count = iteration_count

        self._grid_x_size = grid_x_size
        self._grid_y_size = grid_y_size
        self._activation_map = np.zeros((grid_x_size, grid_y_size))
        self._neigx = np.arange(grid_x_size)
        self._neigy = np.arange(grid_y_size)  # used to evaluate the neighborhood function
        self._output_length = output_length

        if topology not in ['hexagonal', 'rectangular']:
            msg = '%s not supported only hexagonal and rectangular available'
            raise ValueError(msg % topology)

        self.topology = topology
        self._xx, self._yy = np.meshgrid(self._neigx, self._neigy)
        self._xx = self._xx.astype(float)
        self._yy = self._yy.astype(float)

        if topology == 'hexagonal':
            self._xx[::-2] -= 0.5
            if neighborhood_function in ['triangle']:
                warn('triangle neighborhood function does not ' +
                     'take in account hexagonal topology')

        self._decay_function = decay_function

        neig_functions = {'gaussian': self._gaussian,
                          'mexican_hat': self._mexican_hat,
                          'bubble': self._bubble,
                          'triangle': self._triangle}

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function,
                                    ', '.join(neig_functions.keys())))

        if neighborhood_function in ['triangle',
                                     'bubble'] and (divmod(sigma, 1)[1] != 0
                                                    or sigma < 1):
            warn('sigma should be an integer >=1 when triangle or bubble' +
                 'are used as neighborhood function')

        self.neighborhood = neig_functions[neighborhood_function]

        distance_functions = {'euclidean': self._euclidean_distance,
                              'cosine': self._cosine_distance,
                              'manhattan': self._manhattan_distance}

        if activation_distance not in distance_functions:
            msg = '%s not supported. Distances available: %s'
            raise ValueError(msg % (activation_distance,
                                    ', '.join(distance_functions.keys())))

        self._activation_distance = distance_functions[activation_distance]
    
    def fit(self, X):
        self.input_len_ = X.shape[1] - self._output_length
        # random initialization
        self._weights = self._random_generator.rand(self._grid_x_size, self._grid_y_size, X.shape[1]) * 2 - 1
        self._weights /= np.linalg.norm(self._weights, axis=-1, keepdims=True)

        self._check_iteration_number(self._iteration_count)
        #self._check_input_len(X)

        iterations = _build_iteration_indexes(X.shape[0], self._iteration_count,
                                              False, None)

        for t, iteration in enumerate(iterations):
            eta = self._decay_function(self._learning_rate, t, self._iteration_count)
            # sigma and learning rate decrease with the same rule
            sig = self._decay_function(self._sigma, t, self._iteration_count)
            # improves the performances
            g = self.neighborhood(self.winner(X[iteration]), sig)*eta
            # w_new = eta * neighborhood_function * (x-w) for the input part
            self._weights[:, :, :self.input_len_] += np.einsum('ij, ijk->ijk', g,
                                X[iteration][:self.input_len_] - self._weights[:, :, :self.input_len_])
            # w_new = eta * neighborhood_function * (x-w) for the output part
            self._weights[:, :, self._output_length:] += np.einsum('ij, ijk->ijk', g,
                                X[iteration][self._output_length:] - self._weights[:, :, self._output_length:])

        return self

    def _activate(self, x):
        """Updates matrix activation_map, in this matrix
           the element i,j is the response of the neuron i,j to x."""
        self._activation_map = self._activation_distance(x, self._weights[:, :, :self.input_len_])

    def activate(self, x):
        """Returns the activation map to x."""
        self._activate(x)
        return self._activation_map

    def _check_iteration_number(self, num_iteration):
        if num_iteration < 1:
            raise ValueError('num_iteration must be > 1')

    def _check_input_len(self, data):
        """Checks that the data in input is of the correct shape."""
        data_len = len(data[0, :self.input_len_])
        if self.input_len_ != data_len:
            msg = 'Received %d features, expected %d.' % (data_len,
                                                          self.input_len_)
            raise ValueError(msg)

    def winner(self, x):
        """Computes the coordinates of the winning neuron for the sample x."""
        self._activate(x[:self.input_len_])
        return np.unravel_index(self._activation_map.argmin(),
                             self._activation_map.shape)
    
    def _gaussian(self, c, sigma):
        """Returns a Gaussian centered in c."""
        d = 2 * np.pi * sigma * sigma
        ax = np.exp(-np.power(self._xx-self._xx.T[c], 2)/d)
        ay = np.exp(-np.power(self._yy-self._yy.T[c], 2)/d)
        return (ax * ay).T  # the external product gives a matrix

    def _mexican_hat(self, c, sigma):
        """Mexican hat centered in c."""
        p = np.power(self._xx-self._xx.T[c], 2) + np.power(self._yy-self._yy.T[c], 2)
        d = 2*pi*sigma*sigma
        return (np.exp(-p/d)*(1-2/d*p)).T

    def _bubble(self, c, sigma):
        """Constant function centered in c with spread sigma.
        sigma should be an odd value.
        """
        ax = logical_and(self._neigx > c[0]-sigma,
                         self._neigx < c[0]+sigma)
        ay = logical_and(self._neigy > c[1]-sigma,
                         self._neigy < c[1]+sigma)
        return np.outer(ax, ay)*1.

    def _triangle(self, c, sigma):
        """Triangular function centered in c with spread sigma."""
        triangle_x = (-np.abs(c[0] - self._neigx)) + sigma
        triangle_y = (-np.abs(c[1] - self._neigy)) + sigma
        triangle_x[triangle_x < 0] = 0.
        triangle_y[triangle_y < 0] = 0.
        return np.outer(triangle_x, triangle_y)

    def _cosine_distance(self, x, w):
        num = (w * x).sum(axis=2)
        denum = np.multiply(np.linalg.norm(w, axis=2), np.linalg.norm(x))
        return 1 - num / (denum+1e-8)

    def _euclidean_distance(self, x, w):
        return np.linalg.norm(np.subtract(x, w), axis=-1)

    def _manhattan_distance(self, x, w):
        return np.linalg.norm(np.subtract(x, w), ord=1, axis=-1)

    def activation_response(self, data):
        """
            Returns a matrix where the element i,j is the number of times
            that the neuron i,j have been winner.
        """
        #self._check_input_len(data)
        a = np.zeros((self._weights.shape[0], self._weights.shape[1]))
        for x in data:
            a[self.winner(x)] += 1
        return a

if __name__ == "__main__":
    data = np.genfromtxt('examples/iris.csv', delimiter=',', usecols=(0, 1, 2, 3))
    data = data[:, :2] # using only the first two columns

    # data normalization
    data = data -  np.mean(data, axis=0)
    data /= np.std(data)

    # Initialization and training
    vqtam = VQTAM(3, 1, sigma=0.5, learning_rate=0.5, random_seed=10)
    vqtam.fit(data)

    print(vqtam.activation_response(data))