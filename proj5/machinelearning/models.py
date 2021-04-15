import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)
        self.batch_size = 1
        self.multiplier = 1

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        continuing = True
        while (continuing):
            num_wrong = 0
            for x,y in dataset.iterate_once(self.batch_size):
                if self.get_prediction(x) != nn.as_scalar(y):
                    self.w.update(nn.Constant(nn.as_scalar(y)*x.data), self.multiplier)
                    num_wrong += 1
            if num_wrong == 0:
                continuing = False
            
            

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 50
        self.learning_rate = .1
        self.w1 = nn.Parameter(1, 100)
        self.b1 = nn.Parameter(1, 100)
        self.w2 = nn.Parameter(100, 100)
        self.b2 = nn.Parameter(1, 100)
        self.w3 = nn.Parameter(100, 50)
        self.b3 = nn.Parameter(1, 50)
        self.w4 = nn.Parameter(50, 1)
        #need this after reading nn.Gradient documentation
        self.weights = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4]


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        affine = nn.Linear(x, self.w1)
        bias1 = nn.AddBias(affine, self.b1)
        relued = nn.ReLU(bias1)
        affine2 = nn.Linear(relued, self.w2)
        bias2 = nn.AddBias(affine2, self.b2)
        relued2 = nn.ReLU(bias2)
        affine3 = nn.Linear(relued2, self.w3)
        bias3 = nn.AddBias(affine3, self.b3)
        relued3 = nn.ReLU(bias3)
        out = nn.Linear(relued3, self.w4)
        return out



    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_pred = self.run(x)
        return nn.SquareLoss(y_pred, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        #just an arbitrary nonzero loss node
        loss = nn.DotProduct(self.w1, self.w1)

        while nn.as_scalar(loss) >= .00001:
            for x,y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x,y)
                gradients = nn.gradients(loss, self.weights)
                for i in range(len(self.weights)):
                    self.weights[i].update(gradients[i], -self.learning_rate)
    






class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 200
        self.learning_rate = .5
        self.w1 = nn.Parameter(784, 392)
        self.b1 = nn.Parameter(1, 392)
        self.w2 = nn.Parameter(392, 186)
        self.b2 = nn.Parameter(1, 186)
        self.w3 = nn.Parameter(186, 94)
        self.b3 = nn.Parameter(1, 94)
        self.w4 = nn.Parameter(94, 47)
        self.b4 = nn.Parameter(1, 47)
        self.w5 = nn.Parameter(47, 10)
        #need this after reading nn.Gradient documentation
        self.weights = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4, self.w5]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        affine = nn.Linear(x, self.w1)
        bias1 = nn.AddBias(affine, self.b1)
        relued = nn.ReLU(bias1)
        affine2 = nn.Linear(relued, self.w2)
        bias2 = nn.AddBias(affine2, self.b2)
        relued2 = nn.ReLU(bias2)
        affine3 = nn.Linear(relued2, self.w3)
        bias3 = nn.AddBias(affine3, self.b3)
        relued3 = nn.ReLU(bias3)
        affine4 = nn.Linear(relued3, self.w4)
        bias4 = nn.AddBias(affine4, self.b4)
        relued4 = nn.ReLU(bias4)
        out = nn.Linear(relued4, self.w5)

        return out

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        logits = self.run(x)
        loss = nn.SoftmaxLoss(logits, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        val_acc = 0
        batch_number = 0
        while val_acc <= .975:
            for x,y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x,y)
                gradients = nn.gradients(loss, self.weights)
                for i in range(len(self.weights)):
                    #adjust weights with decayed learning rate
                    self.weights[i].update(gradients[i], -self.learning_rate * (.999**batch_number))
            val_acc = dataset.get_validation_accuracy()
            batch_number += 1

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
