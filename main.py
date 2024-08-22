import random
import matplotlib.pyplot as plt
import numpy as np

class Network():

    def __init__(self, structure, weights = []):
        
        self.input = self.baseFunction(random.uniform(-30, 30))
        self.structure = structure
        self.output = None
        self.network = []
        
        if weights == []:
            self.weights = self.initWeights()
        else:
            self.weights = self.inputWeights(weights)

    def initWeights(self):
        num_layers = len(self.structure)

        for i in range(1, num_layers):
            layer = []
            for j in range(self.structure[i]):
                node = []
                for k in range(self.structure[i-1]):
                    weight = [None, random.uniform(0, 1)]
                    node.append(weight)

                layer.append([None, node])
            self.network.append(layer)

    def test(self): 
        num_layers = len(self.structure)

        # Places input into the first node
        self.network[0][0][0] = self.input

        # Takes input of a previous node for the activation of the following nodes
        for layer in range(1, num_layers):
            for node in range(self.structure[i]):
                total = 0
                for j in range(self.structure[i-1]):
                    input = self.network[layer-1][j][0] 
                    for weight in range(self.structure[i-1]):
                        total += input * self.network[layer][node][weight][1] 
                self.network[layer][node][0] = self.ReLU(total)

        # Take final output; last layer, first node, first weight data
        output = self.network[-1][0][0]
        expected = self.baseFunction(self.input)

        result = self.sqDiff(output, expected)
        self.outputResult(result)

    def outputResult(self):
        if self.output:
            return self.output
        else:
            raise AssertionError("Output is not defined.")
        
    @staticmethod            
    def inputWeights(weights):
        pass

    @staticmethod
    def ReLU(x):
        return x if x > 0 else 0
    
    @staticmethod
    def baseFunction(x):
        return x**2

    @staticmethod
    def sqDiff(output, expected):
        return output**2 - expected**2

struct = [1, 3, 1]
generations = 50


if __name__ == '__main__':
    results = []
    best_result = 0
    best_network = 0
    for i in range(generations):

        nn = Network(struct)
        nn.initWeights()
        temp = nn.test()
        results.append(temp)

        if temp > best_result:
            best_network = nn
    
    # Create the x-axis values (0 to generations)
    x_values = np.arange(generations)

    # Plot the neural network results
    plt.plot(x_values, results, label="Neural Network Results", marker='o')

    # Plot the y = x^2 graph
    y_squared = x_values ** 2
    plt.plot(x_values, y_squared, label="y = x^2", linestyle='--')

    # Adding titles and labels
    plt.title("Neural Network Results vs. y = x^2")
    plt.xlabel("Generations")
    plt.ylabel("Values")

    # Show legend
    plt.legend()

    # Display the plot
    plt.show()
    


nn.initWeights()
#nn.train()