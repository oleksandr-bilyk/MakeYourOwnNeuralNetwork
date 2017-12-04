import numpy
import scipy.special

#print(numpy.random.normal(0.0, pow(3, - 0.5), (3, 3)))

class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        #self.wih = numpy.random.normal(0.0, pow(self.hnodes, - 0.5), (self.hnodes, self.inodes))
        self.wih = numpy.array([[-0.59306787,  0.25274925, -0.32602831], [-0.16685239,  0.22542431, -0.36808796], [ 0.81883787,  1.29124618, -0.6584239 ]])
        #self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.who = numpy.array([[ 0.80042512,  0.35423876,  0.16241759], [-1.52660991, -0.82271924,  0.23120044], [ 0.66190966,  0.31868365,  0.39380777]])
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        #calculate signals into hidden layer
        hiddenInput = numpy.dot(self.wih, inputs)
        #calculate the signals emerging from hidden layer
        hiddenOutput = self.activation_function(hiddenInput)
        
        #calculate signals into final output layer
        finalInput = numpy.dot(self.who, hiddenOutput)
        #calculate the signals emerging from final output layer
        finalOutput = self.activation_function(finalInput)
        
        #error is the (target - actual)
        output_errors = targets - finalOutput
        # hidden layer errorr is the output_errors, split by weights, recombines at hidden hodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        # update the weights for the links between the hidden and output layers
        left = (output_errors * finalOutput * (1 - finalOutput))
        whoDelta = self.lr * numpy.dot((output_errors * finalOutput * (1 - finalOutput)), numpy.transpose(hiddenOutput))
        self.who += self.lr * numpy.dot((output_errors * finalOutput * (1 - finalOutput)), numpy.transpose(hiddenOutput))
        # update the weights for the links between the input and hidde layers
        wihDelta = self.lr * numpy.dot((hidden_errors * hiddenOutput * (1- hiddenOutput)), numpy.transpose(inputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hiddenOutput * (1- hiddenOutput)), numpy.transpose(inputs))
        pass
    
    def query(self, inputs_list):
        #convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        #calculate signals into hidden layer
        hiddenInput = numpy.dot(self.wih, inputs)
        #calculate the signals emerging from hidden layer
        hiddenOutput = self.activation_function(hiddenInput)
        
        #calculate signals into final output layer
        finalInput = numpy.dot(self.who, hiddenOutput)
        #calculate the signals emerging from final output layer
        finalOutput = self.activation_function(finalInput)
        
        return finalOutput

input_nodes = 3
hidden_nodes = 3
output_nodes = 3
learning_rate = 0.3
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
#firstQuery = n.query([1.0, 0.5, -1.5])
n.train([1.0, 0.5, -1.5], [1.0, 0.5, -1.5])