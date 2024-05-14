import numpy as np


class SOM:
    def __init__(self, input_dimensions, output_dimensions, alpha=0.1, sigma=1.0):
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.alpha = alpha
        self.sigma = sigma
        self.weights = np.random.rand(output_dimensions[0], output_dimensions[1], input_dimensions)
        self.qError = 0

    def train(self, data, epochs):
        for epoch in range(epochs):
            for x in data:
                best_matching_unit = self.find_best_matching_unit(x)
                self.update_weights(x, best_matching_unit, epoch, len(data))

    def find_best_matching_unit(self, x):
        min_dist = float('inf')
        best_matching_unit = np.array([0, 0])
        for i in range(self.output_dimensions[0]):
            for j in range(self.output_dimensions[1]):
                dist = np.linalg.norm(x - self.weights[i, j])
                if dist < min_dist:
                    min_dist = dist
                    best_matching_unit = np.array([i, j])
        self.qError += min_dist
        return best_matching_unit

    def update_weights(self, x, best_matching_unit, epoch, num_samples):
        lr = self.alpha * (1 - epoch / num_samples)
        sigma = self.sigma * (1 - epoch / num_samples)
        for i in range(self.output_dimensions[0]):
            for j in range(self.output_dimensions[1]):
                dist = np.linalg.norm(best_matching_unit - np.array([i, j]))
                influence = np.exp(-dist ** 2 / (2 * sigma ** 2))
                self.weights[i, j] += lr * influence * (x - self.weights[i, j])


# Usage example
data = np.random.rand(100, 3)   # 100 samples with three features (RGB colors)
som = SOM(input_dimensions=3, output_dimensions=(5, 5))
som.train(data, epochs=100)

# Creating a color map based on the order of input samples
color_map = np.empty((5, 5, 3))
for i, x in enumerate(data):
    bmu = som.find_best_matching_unit(x)
    color_map[bmu[0], bmu[1]] = x

# Plotting the color map
import matplotlib.pyplot as plt

plt.imshow(color_map)
plt.title('SOM Color Map')
plt.axis('off')
plt.show()
