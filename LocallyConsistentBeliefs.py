import os
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import math
import random
from tensorflow import python
from tensorflow.python.keras.backend import eager_learning_phase_scope
import mlflow

#matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
np.random.seed(3120)
random.seed(6)
tf.random.set_seed(15)




def noiser(x, y, eps):
    """
    Adds  i.i.d. distributed Gaussian noise on every component of y. Returns x and noised version of y.
    """
    n = y.size
    noi = np.random.normal(0, eps, n)
    y = y + noi
    return (x, y)


def sinplotter2(d, freq):
    """
    Generates a data sample of size d, whose input points are equidistant on [pi/4,2pi], but mixed and output points are
    y_i= x_i *sin(freq*x_i)/6
    """
    x = np.linspace(np.pi / 4, 2 * np.pi, d)
    random.shuffle(x)
    y = x * np.sin(freq * x) / 6
    return (x, y)


def sinplotter3(d, freq):
    """
    Generates a data sample of size d, whose input points are equidistant on [pi/4,2pi] and output points are
    y_i= x_i *sin(freq*x_i)/6.
    """
    x = np.linspace(2, 6, d)
    y = x * np.sin(freq * x) / 6
    return (x, y)


def otherfunction(d):
    """
    Generates a data sample of size d, where 1/4 of input points are equidistant on [1.75,3], 3/4 of input points are
    equidistant on [4,7], but mixed and output points are y_i= x_i^2 *exp(-(x-4)^2)/16.
    """
    print(int(d/4))
    x =np.concatenate( (np.linspace(1.75,3, int(d/4)),np.linspace(4,7, int(3/4*d))))
    random.shuffle(x)
    y = x ** 2 * np.exp(-(x - 4) ** 2) / 16
    return (x, y)


def otherfunction2(d):
    """
    Generates a data sample of size d, where 1/4 of input points are equidistant on [1.75,3], 3/4 of input points are
    equidistant on [4,7] and output points are y_i= x_i^2 *exp(-(x-4)^2)/16.
    """
    x = np.linspace(1.5 , 7.5, d)
    y = x ** 2 * np.exp(-(x - 4) ** 2) / 16
    return (x, y)

def TriangleDistributedSinus(a,b,c,n):
    """
    Generates a data sample of size n, where the input points are triangular distributed on [a,c] with mode at b
    and output points are y_i= sin(x_i/6).
    """
    x=np.random.triangular(a,b,c,n)
    y= np.sin(x/6)
    return(x,y)

def NotrandomSinus(a,c,n):
    """
    Generates a data sample of size n, where the inputs x_i are equidistant on [a,c] and the output is $y_i=sin(x_i/6)
    """
    x = np.linspace(a,c,n )
    y = np.sin(x/6)
    return (x, y)

def datareshape(x):
    """
    Converts a nd.aaray x of shape (n,...) into a tensor of shape (n,1).
    """
    x=np.reshape(x, (x.shape[0],1))
    x=tf.convert_to_tensor(x)
    return (x)

class Dropout(tf.keras.layers.Dropout):
    """
    Auxilary class to generate a Model that applies Dropout in both training and testing.
    This is copied from Alexander Herzog (https://github.com/keras-team/keras/issues/9412).
    """
    def __init__(self, rate, training=None, noise_shape=None, seed=None, **kwargs):
        super(Dropout, self).__init__(rate, noise_shape=None, seed=None, **kwargs)
        self.training = training

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return tf.keras.backend.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed)

            if not training:
                return tf.keras.backend.in_train_phase(dropped_inputs, inputs, training=self.training)
            return tf.keras.backend.in_train_phase(dropped_inputs, inputs, training=training)
        return inputs

class f:
    """
    Class of Deep Neural Networks
    """
    def __init__(self,n ,alpha,sigma,x,y,i,activation):
        """
        Initialize a neural network with
        :param n: size of data sample
        :param alpha: magnititude of \alpha in the regularizing loss L_\alpha
        :param sigma: Variance of homeostatic noise used to create data sample.
        :param x: inputs of data sample for training
        :param y: outputs of data sample for training
        :param i: in 0,1,2 decides the network architecture. In case of i, the architecture is initialized with buildi
        :param activation: Activation function as string.
        """
        self.optimizer = tf.keras.optimizers.Adam(0.001)
        self.loss = 'mean_squared_error'
        self.units = 16
        self.activation = activation
        self.alpha = alpha
        self.models = [self.build0(),self.build1(),self.build2()]
        self.model=self.models[i]
        self.m=self.model.count_params()
        self.x=x
        self.y=y
        self.n=n
        self.sigma=sigma

    def build0(self):
        """
        Builds a tf.keras model with 4 layers, 16 units each where the activation and alpha is specified by attributes.
        Without dropout.
        """
        model = tf.keras.Sequential([tf.keras.layers.Dense(self.units,
                                                           kernel_regularizer=tf.keras.regularizers.l2(l=self.alpha),
                                                           name="fc1", activation=self.activation, input_shape=[1]),
                                     tf.keras.layers.Dense(self.units,
                                                           kernel_regularizer=tf.keras.regularizers.l2(l=self.alpha),
                                                           name="fc2", activation=self.activation),
                                     tf.keras.layers.Dense(self.units,
                                                           kernel_regularizer=tf.keras.regularizers.l2(l=self.alpha),
                                                           name="fc3", activation=self.activation),
                                     tf.keras.layers.Dense(self.units,
                                                           kernel_regularizer=tf.keras.regularizers.l2(l=self.alpha),
                                                           name="fc4", activation=self.activation),
                                     tf.keras.layers.Dense(1)])
        model.compile(loss=self.loss, optimizer=self.optimizer)
        model.summary()
        return model

    def build2(self):
        """
        Builds a tf.keras model with 4 layers, 16 units each where the activation and alpha is specified by attributes.
        With dropout.
        """
        model = tf.keras.Sequential([tf.keras.layers.Dense(self.units,
                                                           kernel_regularizer=tf.keras.regularizers.l2(l=self.alpha),
                                                           name="fc1", activation=self.activation, input_shape=[1]),
                                     Dropout(0.15,training=True),
                                     tf.keras.layers.Dense(self.units,
                                                           kernel_regularizer=tf.keras.regularizers.l2(l=self.alpha),
                                                           name="fc2", activation=self.activation),
                                     Dropout(0.15,training=True),
                                     tf.keras.layers.Dense(self.units,
                                                           kernel_regularizer=tf.keras.regularizers.l2(l=self.alpha),
                                                           name="fc3", activation=self.activation),
                                     Dropout(0.15,training=True),
                                     tf.keras.layers.Dense(self.units,
                                                           kernel_regularizer=tf.keras.regularizers.l2(l=self.alpha),
                                                           name="fc4", activation=self.activation),
                                     Dropout(0.15,training=True),
                                     tf.keras.layers.Dense(1)])
        model.compile(loss=self.loss, optimizer=self.optimizer)
        model.summary()
        return model


    def build1(self):
        """
        Builds a tf.keras model with 2 layers, 16 units each where the activation and alpha is specified by attributes.
        Without dropout.
        """
        model = tf.keras.Sequential([tf.keras.layers.Dense(self.units,
                                                           kernel_regularizer=tf.keras.regularizers.l2(l=self.alpha),
                                                           name="fc1", activation=self.activation, input_shape=[1]),
                                     tf.keras.layers.Dense(self.units,
                                                           kernel_regularizer=tf.keras.regularizers.l2(l=self.alpha),
                                                           name="fc2", activation=self.activation)
                                        , tf.keras.layers.Dense(1)
                                     ])
        model.compile(loss=self.loss, optimizer=self.optimizer)
        model.summary()
        return model

    def train(self, epochs, batch_size):
        """
        Trains the model with epochs=epochs and batch_size=batch_size and the data that is specified as attributes.
        """
        self.model.fit(self.x, self.y, epochs=epochs, batch_size=batch_size)

    def test(self, x, y,x_test,y_test):
        """
        Produces a Plot with training data as dots and the predictions at the x_test s as line
        :param x: numpy array of training inputs
        :param y: numpy array of training outputs
        :param x_test: numpy array of test inputs
        :param y_test: numpy array of test outputs
        """
        prediction = self.model.predict(x_test)
        plt.figure(figsize=(5.8, 3.0))
        plt.scatter(x, y, marker=".", color="#95d0fc")
        plt.plot(x_test, prediction, "#001146")
        plt.title(r'$x\mapsto g_{\theta(\mathbf{d})}(x)$')
        #plt.savefig('Modelfit5.pgf')
        plt.show()

    def covfunctiontest(self,  x_1, x_2, x_3, x_4, x, y):
        """
        Produces a four plots of the covariance function $K_d(x_1,x)$ derived in the thesis.
        :param x_1: singular input value (float)
        :param x_2: singular input value (float)
        :param x_3:singular input value (float)
        :param x_4: singular input value (float)
        :param x: numpy array of input values
        """
        plt.figure(figsize=(5.8, 3.0))
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
        covf1 = self.sigma/ self.alpha /self.n * np.array([self.covariance(x_1, x_0) for x_0 in x])
        covf2 = self.sigma/ self.alpha /self.n  * np.array([self.covariance(x_2, x_0) for x_0 in x])
        covf3 = self.sigma/ self.alpha /self.n  * np.array([self.covariance(x_3, x_0) for x_0 in x])
        covf4 = self.sigma/ self.alpha /self.n  * np.array([self.covariance(x_4, x_0) for x_0 in x])
        axs[0, 0].plot(x, covf1, "#0343df")
        axs[0, 1].plot(x, covf2, "#0343df")
        axs[1, 0].plot(x, covf3, "#0343df")
        axs[1, 1].plot(x, covf4, "#0343df")
        axs[0, 0].set_title(r'$x\mapsto \sigma_{\epsilon}^2k_{\mathbf{d}}(1.2,x)$')
        axs[0, 1].set_title(r'$x\mapsto \sigma_{\epsilon}^2k_{\mathbf{d}}(2.6,x)$')
        axs[1, 0].set_title(r'$x\mapsto \sigma_{\epsilon}^2k_{\mathbf{d}}(3.8,x)$')
        axs[1, 1].set_title(r'$x\mapsto \sigma_{\epsilon}^2k_{\mathbf{d}}(5.1,x)$')
        #plt.savefig('Covariancefunction5.pgf')
        plt.show()


    def getOtherHalfHessian(self):
        """
        :return: Returns the approximation of the Hessian of the loss.
        """
        A= np.array([self.getgradient(x_1) for x_1 in self.x])
        print(A)
        print(A.shape)
        return(np.dot(np.transpose(A),A))

    def getInverseHalfHessian(self):
        """
        :return: Returns the inverse of the approximation of the Hessian of the loss.
        """
        HalfHessian=np.array(self.get_H_op())-self.getOtherHalfHessian()+alpha *np.eye(self.m,self.m)
        return(np.linalg.inv(HalfHessian))


    def getgradient(self, x_1):
        """
        :param x_1: singular input value (float)
        :return: gradient of the loss at x_1
        """
        x_1 = x_1 * tf.ones((1, 1))
        with tf.GradientTape() as tape:
            pred1 = self.model(x_1)[0]
        help = self.flatten(tape.gradient(pred1, self.model.trainable_variables))
        return (np.array(help))


    def covariance(self, x_1, x_2):
        """
        :param x_1: singular input value (float)
        :param x_2: singular input value (float)
        :return: Covariance $K_d(x_1,x_2)$ derived in the thesis
        """
        return (np.dot(self.getgradient(x_1), self.getgradient(x_2)))

    def flatten(self, params):
        """
        The following function is written by Geir Nilsen (https://github.com/gknilsen/pyhessian).
        Flattens the list of tensor(s) into a 1D tensor
        Args:
            params: List of model parameters (List of tensor(s))
        Returns:
            A flattened 1D tensor
        """
        return tf.concat([tf.reshape(_params, [-1]) \
                          for _params in params], axis=0)

    def prediction(self,x):
        """
        :param x: Numpy array of input values
        :return: Numpy array of prediction from the Neural network
        """
        return (np.reshape(self.model.predict(x), x.shape))

    def get_Hv_op(self, v):
        """
        Adapted function from Geir Nilsen (https://github.com/gknilsen/pyhessian). This is a version that also performs
        in eager mode.

        Implements a Hessian vector product estimator Hv op defined as the
        matrix multiplication of the Hessian matrix H with the vector v.
        :param v: Vector to multiply with Hessian (tensor)
        :return: Hessian vector product op (tensor)
        """
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
            #self.cost = tf.keras.losses.MSE(y,self.prediction(x))
                self.cost= tf.math.reduce_mean(tf.keras.losses.MSE(datareshape(self.y),sine_model.model(datareshape(self.x))))
            cost_gradient = self.flatten(tape2.gradient(self.cost,
                                                  self.model.trainable_variables))
            vprod = tf.math.multiply(cost_gradient,
                                 tf.stop_gradient(v))
            Hv_op = self.flatten(tape1.gradient(vprod,
                                          self.model.trainable_variables))
        return Hv_op

    def get_H_op(self):
        """
        Adapted from Geir Nilsen(https://github.com/gknilsen/pyhessian):
        Implements a full Hessian estimator op by forming m Hessian vector
        products using HessianEstimator.get_Hv_op(v) for all v's in R^P

        Returns:
            H_op: Hessian matrix op (tensor)
        """
        H_op = tf.map_fn(self.get_Hv_op, tf.eye(self.m, self.m), dtype='float32')
        return H_op


    def getInverseHessian(self):
        """
        :return: Inverse of the Hessian of the loss
        """
        H = np.array(self.getOtherHalfHessian())
        #H=np.array(self.get_H_op())
        H=H+self.n *self.alpha * np.eye(self.m,self.m)
        return(np.linalg.inv(H))

    def PointUncertaintyEstimate(self,Hinverse, xi):
        """
        :param Hinverse: Inverse of the Hessian of the loss
        :param xi: singular input value
        :return: Estimate of hessian following the theorem 4.9 in the thesis
        """
        kd=self.getgradient(xi)
        PU=np.dot(kd, np.matmul(Hinverse,kd))
        return (self.sigma*PU)

    def UncertaintyEstimate(self,x):
        """
        :param x: Numpy array of input values
        :return: Numpy array of uncertainty estimates in theorem 4.9 in the thesis
        """
        Hinverse=self.getInverseHessian()
        return(np.array([ np.sqrt(self.PointUncertaintyEstimate(Hinverse, xi)) for xi in x]))

    def Uncertaintytest(self, x,y,x_test,y_test):
        """
        Produces a Plot of the training input as dots, the Neural network fit as dark line and the fit plus minus
        the standard derivation following the thesis as blue line.
        :param x: training input data
        :param y: training output data
        :param x_test: test input data
        """
        prediction =self.model.predict(x_test)
        Uncertainty= self.UncertaintyEstimate(x_test)
        #Uncertainty=np.transpose(Uncertainty)
        prediction=np.transpose(prediction)[0]
        plt.figure(figsize=(5.8, 3.0))
        plt.scatter(x, y, marker=".", color="#95d0fc")
        plt.plot(x_test, prediction, "#001146")
        plt.plot(x_test, np.add(prediction,Uncertainty), "#0343df")
        plt.plot(x_test, np.add(prediction,-Uncertainty), "#0343df")
        #plt.title(r'$x^\star\mapsto U(x^\star)$')
        #plt.savefig('Comparison-LocBel.pgf')
        plt.show()



    def predict_with_uncertainty(self, x, n_iter=500):
        """
        This function provides the prediction and uncertainty of the neural network when using Dropout.
        :param x: numpy array of inputs
        :param n_iter: sample size
        :return prediction: the sample mean of all n_iter predictions
        :return uncertainty: the sample standard derivation of n_iter predictions
        """
        result=[]
        for i in range(n_iter):
            result.append(self.model.predict(x))
        result=np.array(result)
        prediction = result.mean(axis=0)
        uncertainty = result.std(axis=0)
        return prediction, uncertainty


    def testwithdropout(self, x, y,x_test,y_test):
        """
        Produces Plot of training inputs as dots, plot of prediction via Dropout and prediction plus minus standard
        derivation by using Dropout.
        :param x: numpy array of training inputs
        :param y: numpy array of training outpus
        :param x_test: numpy array of test inputs
        """
        prediction,uncertainty = self.predict_with_uncertainty(x_test)
        plt.figure(figsize=(5.8, 3.0))
        plt.scatter(x, y, marker=".", color="#95d0fc")
        plt.plot(x_test, prediction, "#001146")
        plt.plot(x_test, prediction+uncertainty, "#0343df")
        plt.plot(x_test,prediction-uncertainty, "#0343df")
        #plt.savefig('Comparison-Dropout.pgf')
        plt.show()





if __name__ == "__main__":

    #hyperparameters
    #alpha = 0.01
    alpha=0.0001
    alpha2 =0.00001
    Batch_size=64

    #n = 12000
    n=200
    #n=30
    sigma = 0.02
    activation="relu"
    #activation="tanh"


    # create data
    """
    x, y = sinplotter2(n, 2)
    x, y = noiser(x, y, sigma)
    x_test, y_test = sinplotter3(1000, 2)
    x_test, y_test = noiser(x_test, y_test, sigma)
    """
    """
    # create alternative data
    x, y = TriangleDistributedSinus(0,4,30,n)
    x, y = noiser(x, y, sigma)
    x_test, y_test = NotrandomSinus(-10,40, 1000)
    x_test, y_test = noiser(x_test, y_test, sigma)
    """
    # create alternativealernative data
    x, y = otherfunction( n)
    x, y = noiser(x, y, sigma)
    x_test, y_test = otherfunction2(1000)
    x_test, y_test = noiser(x_test, y_test, sigma)

    # create model
    sine_model = f(n, alpha,sigma,x,y,0, activation)
    sine_model.train(1000, 16)
    #sine_model2 = f(n, alpha2, sigma, x, y, 2, activation)
    #sine_model2.train(1000, 16)

    # create Plots
    #sine_model.test(x,y,x_test, y_test)
    #sine_model2.testwithdropout(x,y,x_test,y_test)
    sine_model.Uncertaintytest(x,y, x_test,y_test)
    #sine_model.covfunctiontest(1.2, 2.6, 3.8, 5.1, x_test ,y_test)






