import numpy as np


class MSOM:
    def __init__(self, input_dimensions, output_dimensions, context_dimensions, alpha=0.1, beta=0.1, sigma=1.0):
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.context_dimensions = context_dimensions
        self.alpha = alpha  # learning_rate
        self.beta = beta  # context_rate
        self.sigma = sigma
        self.weights = np.random.rand(output_dimensions[0], output_dimensions[1], input_dimensions)
        self.context_weights = np.random.rand(output_dimensions[0], output_dimensions[1], context_dimensions)
        self.qError = 0

    def train(self, data, epochs):
        for epoch in range(epochs):
            for seq in data:
                for t, x in enumerate(seq):
                    self.update_context(seq[:t])
                    best_matching_unit = self.find_best_matching_unit(x)
                    self.update_weights(x, best_matching_unit, epoch, len(data))

    def find_best_matching_unit(self, x):
        min_dist = float('inf')
        best_matching_unit = np.array([0, 0])
        for i in range(self.output_dimensions[0]):
            for j in range(self.output_dimensions[1]):
                dist = self.calculate_distance(x, i, j)
                if dist < min_dist:
                    min_dist = dist
                    best_matching_unit = np.array([i, j])
        self.qError += min_dist
        return best_matching_unit

    def update_weights(self, x, best_matching_unit, epoch, num_samples):
        theta = self.alpha * (1 - epoch / num_samples)
        sigma = self.sigma * (1 - epoch / num_samples)
        for i in range(self.output_dimensions[0]):
            for j in range(self.output_dimensions[1]):
                dist = np.linalg.norm(best_matching_unit - np.array([i, j]))
                gamma = np.exp(-dist ** 2 / (2 * sigma ** 2))  # influence
                self.weights[i, j] += theta * gamma * (x - self.weights[i, j])
                self.context_weights[i, j] += theta * gamma * (x - self.context_weights[i, j])

    def update_context(self, seq):
        if len(seq) > 0:
            context = np.mean(seq, axis=0)
            self.context_weights = (1 - self.beta) * self.weights + self.beta * context

    def calculate_distance(self, x, i, j):
        input_weight = self.weights[i, j]
        context_weight = self.context_weights[i, j]
        return (1 - self.alpha) * np.linalg.norm(x - input_weight) + self.alpha * np.linalg.norm(
            context_weight - input_weight)


# Usage example
data = np.random.rand(100, 3, 3)
#100 samples with three features each (representing RGB colors) arranged in sequences of three colors.
msom = MSOM(input_dimensions=3, output_dimensions=(5, 5), context_dimensions=3)
msom.train(data, epochs=100)

# Creating a color map based on the order of input samples
color_map = np.empty((5, 5, 3))
for i, x in enumerate(data):
    bmu = msom.find_best_matching_unit(x)
    color_map[bmu[0], bmu[1]] = x[2] #2 means that only the end of the sequence is displayed

# Plotting the color map
import matplotlib.pyplot as plt

plt.imshow(color_map)
plt.title('MSOM Color Map')
plt.axis('off')
plt.show()
