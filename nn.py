import random
import matplotlib.pyplot as plt
import numpy as np
import math
import copy

class Network:

    def __init__(self, structure, weights=[]):
        self.structure = structure
        self.output = None
        self.network = []
        self.biases = []  # Add biases
        self.input_weights = []  # Add input weights

        if weights == []:
            self.initWeights()
        else:
            self.inputWeights(weights)

    def initWeights(self):
        num_layers = len(self.structure)
        
        # Initialize input weights for inputs
        self.input_weights = [random.uniform(-0.1, 0.1) for _ in range(self.structure[0])]

        for i in range(1, num_layers):
            layer = []
            bias_layer = []
            for j in range(self.structure[i]):
                node = []
                for k in range(self.structure[i-1]):
                    weight = random.uniform(-0.1, 0.1)  # Smaller weight range
                    node.append(weight)
                layer.append([None, node])
                bias_layer.append(random.uniform(-0.1, 0.1))  # Initialize biases
            self.network.append(layer)
            self.biases.append(bias_layer)  # Add biases per layer

        # Appends 0 as the network input to be changed during evaluation
        input_layer = []
        for _ in range(self.structure[0]):
            input_layer.append([0])
        self.network.insert(0, input_layer)

    def modifyWeights(self):
        num_layers = len(self.structure)
        for i in range(1, num_layers):
            for j in range(self.structure[i]):
                for k in range(self.structure[i-1]):
                    delta = random.uniform(-0.05, 0.05)
                    self.network[i][j][1][k] += delta
                self.biases[i-1][j] += random.uniform(-0.05, 0.05)  # Modify biases

        # Modify input weights
        for i in range(len(self.input_weights)):
            self.input_weights[i] += random.uniform(-0.05, 0.05)

    def evaluate(self, x): 
        num_layers = len(self.structure)
        
        # Multiply input by input weights
        weighted_input = [x[i] * self.input_weights[i] for i in range(len(x))]
        self.network[0] = [[val] for val in weighted_input]  # Place weighted input into the first nodes

        for layer in range(1, num_layers):
            curr_layer_nodes = self.structure[layer]
            prev_layer_nodes = self.structure[layer-1]

            for i in range(curr_layer_nodes):
                total = self.biases[layer-1][i]  # Start with the bias
                for j in range(prev_layer_nodes):
                    weight = self.network[layer][i][1][j]
                    activation = self.network[layer-1][j][0]
                    total += weight * activation
                self.network[layer][i][0] = self.tanh(total)  # Using tanh for better representation

        output = self.network[-1][0][0]
        expected = self.baseFunction(x)  # Multivariable base function

        result = self.sqDiff(output, expected)
        self.output = result

        return self.outputResult()

    def backpropagate(self, x, learning_rate=0.01):
        # Forward pass
        self.evaluate(x)
        output = self.network[-1][0][0]
        expected = self.baseFunction(x)

        # Calculate the error
        error = output - expected

        # Backward pass for layers
        for layer in reversed(range(1, len(self.structure))):
            for i in range(self.structure[layer]):
                if layer == len(self.structure) - 1:
                    delta = error * self.tanh_derivative(self.network[layer][i][0])
                else:
                    delta = sum([self.network[layer+1][k][1][i] * self.network[layer+1][k][2] for k in range(self.structure[layer+1])])
                    delta *= self.tanh_derivative(self.network[layer][i][0])
                self.network[layer][i].append(delta)

                for j in range(self.structure[layer-1]):
                    weight_update = learning_rate * delta * self.network[layer-1][j][0]
                    self.network[layer][i][1][j] -= weight_update

                self.biases[layer-1][i] -= learning_rate * delta  # Update biases

        # Backward pass for input weights
        for i in range(len(self.input_weights)):
            self.input_weights[i] -= learning_rate * error * x[i]  # Adjust input weights

    def addNode(self):
        """
        Adds a node to a specific layer.
        """
        layer_index = random.randrange(2, len(self.structure))
        
        # Add a new node
        self.structure[layer_index] += 1
        new_node_weights = [random.uniform(-0.1, 0.1) for _ in range(self.structure[layer_index-1])]
        new_bias = random.uniform(-0.1, 0.1)
        self.network[layer_index].append([None, new_node_weights])
        self.biases[layer_index-1].append(new_bias)
        
        # Update next layer weights to include new node
        if layer_index < len(self.structure) - 1:
            for node in self.network[layer_index + 1]:
                node[1].append(random.uniform(-0.1, 0.1))

    def addLayer(self):
        """
        Adds a new layer at the specified index with a given number of nodes.
        """

        layer_index = random.randrange(3, len(self.structure))

        if layer_index < 1 or layer_index >= len(self.structure):
            raise ValueError("Cannot add a layer at this index.")
        
        # Add new layer
        num_nodes = random.randrange(5, 9)

        self.structure.insert(layer_index, num_nodes)
        new_layer = []
        new_biases = []
        for _ in range(num_nodes):
            new_node_weights = [random.uniform(-0.1, 0.1) for _ in range(self.structure[layer_index-1])]
            new_layer.append([None, new_node_weights])
            new_biases.append(random.uniform(-0.1, 0.1))
        self.network.insert(layer_index, new_layer)
        self.biases.insert(layer_index-1, new_biases)

        # Update next layer weights to include new nodes
        for node in self.network[layer_index + 1]:
            node[1] = [random.uniform(-0.1, 0.1) for _ in range(num_nodes)]

    def removeNode(self):
        """
        Randomly removes a node from any layer except the input or output layers.
        """
        layer_index = random.randint(1, len(self.structure) - 2)
        if self.structure[layer_index] <= 1:
            return  # Cannot remove node if only one is left

        node_index = random.randint(0, self.structure[layer_index] - 1)
        self.structure[layer_index] -= 1
        del self.network[layer_index][node_index]
        del self.biases[layer_index-1][node_index]

        # Update next layer weights
        for node in self.network[layer_index + 1]:
            del node[1][node_index]

    def outputResult(self):
        if self.output is not None:
            return self.output
        else:
            raise AssertionError("Output is not defined.")
        
    @staticmethod            
    def inputWeights(weights):
        pass

    @staticmethod
    def tanh(x):
        return math.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - math.tanh(x) ** 2

    @staticmethod
    def baseFunction(inputs):
        return 0.5 * inputs[0] + 0.3 * inputs[1] - 0.2 * inputs[2] + 0.4 * inputs[3]

    @staticmethod
    def sqDiff(output, expected):
        return (output - expected)**2

struct = [4, 9, 9, 1]  # Network structure with four inputs
generations = 10
population = 5

if __name__ == '__main__':
    results = []
    lowest_sq_diff = float('inf')
    best_network = Network(struct)

    for i in range(generations):
        for j in range(population):
            nn = copy.deepcopy(best_network)
            for _ in range(50):  # Training with 50 samples per generation
                x = [random.uniform(-5, 5) for _ in range(4)]  # Generate 4 random inputs
                nn.backpropagate(x, learning_rate=0.05)  # Increased learning rate

                # Randomly decide to add a node or layer
                if random.random() < 0.1:  # 10% chance to add a node
                    nn.addNode()
                if random.random() < 0.05:  # 5% chance to add a layer
                    nn.addLayer()

                # Randomly decide to remove a node
                if random.random() < 0.1:  # 10% chance to remove a node
                    nn.removeNode()

            curr_sq_diff = 0
            checks = 50
            for _ in range(checks):
                x = [random.uniform(-5, 5) for _ in range(4)]
                curr_sq_diff += nn.evaluate(x)
            curr_sq_diff /= checks

            print(f"Generation {i+1}, Individual {j+1}, Squared Difference: {curr_sq_diff}")

            if curr_sq_diff < lowest_sq_diff:
                lowest_sq_diff = curr_sq_diff
                best_network = nn
                results.append(curr_sq_diff)
    
    # Plot the values of the input multipliers (input weights) after training
    plt.bar(range(len(best_network.input_weights)), best_network.input_weights)
    plt.xlabel("Input Index")
    plt.ylabel("Input Multiplier (Weight)")
    plt.title("Input Multipliers After Training")
    plt.show()
