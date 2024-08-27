import random
import matplotlib.pyplot as plt
import numpy as np
import math
import copy

class Network():

    def __init__(self, structure, weights = []):
        
        self.input = self.baseFunction(random.uniform(-5, 5))
        self.structure = structure
        self.output = None
        self.network = []
        
        if weights == []:
            self.weights = self.initWeights()
        else:
            self.weights = self.inputWeights(weights)

    def initWeights(self):
        """
        Initializes the neural network by setting its basic structure and random numbers.
        The network is built upon the self.structure variable.
        """
        num_layers = len(self.structure)

        for i in range(1, num_layers):
            layer = []
            for j in range(self.structure[i]):
                node = []
                for k in range(self.structure[i-1]):
                    weight = random.uniform(-1, 1)
                    node.append(weight)

                layer.append([None, node])
            self.network.append(layer)

        # Appends 0 as the network input to be changed during evaluation
        input = []
        input.append([0])
        self.network.insert(0, input)

    def modifyWeights(self):
        """
        Randomly changes the weights of the network by some delta factor
        """
        num_layers = len(self.structure)

        for i in range(1, num_layers):
            for j in range(self.structure[i]):
                for k in range(self.structure[i-1]):
                    delta = random.uniform(-0.5, 0.5)
                    self.network[i][j][1][k] += delta
    
    def evaluate(self, x): 
        num_layers = len(self.structure)

        # Places input into the first node
        input = []
        input.append([x])
        self.network[0] = input

        # Takes input of a previous node for the activation of the following nodes
        for layer in range(1, num_layers):
            curr_layer_nodes = self.structure[layer]
            prev_layer_nodes = self.structure[layer-1]

            for i in range(curr_layer_nodes):
                total = 0
                for j in range(prev_layer_nodes):
                    weight = self.network[layer][i][1][j]
                    activation = self.network[layer-1][j][0] # Activation of the previous node
                    total += weight * activation
                self.network[layer][i][0] = self.ReLU(total) # New-found activation
                

        # Take final output; last layer, first node, first weight data
        output = self.network[-1][0][0]
        expected = self.baseFunction(self.input)

        result = self.sqDiff(output, expected)
        self.output = result

        return self.outputResult()

    def outputResult(self):
        if self.output:
            return self.sqDiff(self.output, self.baseFunction(self.input))
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
        return 5*math.sin(x) - 7*math.cos(x) - 3*math.sin(x)

    @staticmethod
    def sqDiff(output, expected):
        return (output - expected)**2

struct = [1, 3, 3, 3, 3, 3, 1]
generations = 10
population = 5

if __name__ == '__main__':
    results = []
    lowest_sq_diff = float('inf')
    best_network = Network(struct)
    
    for i in range(generations):
        for j in range(population):
            # Make a copy of the best network to try modifications
            nn = copy.deepcopy(best_network)
            nn.modifyWeights()

            curr_sq_diff = 0
            checks = 50
            for _ in range(checks):
                x = random.uniform(-5, 5)
                curr_sq_diff += nn.evaluate(x)
            curr_sq_diff /= checks

            print(f"Generation {i+1}, Individual {j+1}, Squared Difference: {curr_sq_diff}")

            # Update the best network only if the current one is better
            if curr_sq_diff < lowest_sq_diff:
                lowest_sq_diff = curr_sq_diff
                best_network = nn
                results.append(curr_sq_diff)  # Append only the better results
    
    # Create the x-axis values (0 to number of improvements)
    x_values = np.arange(len(results))

    # Plot the neural network results
    print(results)
    plt.plot(x_values, results, label="Neural Network Improvements", marker='o')

    # Adding titles and labels
    plt.title("Square Difference Improvements Over Generations")
    plt.xlabel("Improvement Instances")
    plt.ylabel("Squared Difference")

    # Show legend
    plt.legend()

    # Display the plot
    plt.show()
