import random
import math

class NeuralNetwork:
    def __init__(self):
        # Initialize with Xavier/Glorot initialization
        self.weight = random.uniform(-1/math.sqrt(1), 1/math.sqrt(1))
        self.bias = 0  # Initialize bias to 0
        self.x_min = float('inf')
        self.x_max = float('-inf')

    def normalize_input(self, x):
        """Normalize input to [0,1] range"""
        return (x - self.x_min) / (self.x_max - self.x_min) if self.x_max > self.x_min else 0

    def forward(self, x: float):
        # Normalize input and calculate weighted sum
        x_norm = self.normalize_input(x)
        z = self.weight * x_norm + self.bias
        output = self.sigmoid(z)
        return output

    def predict(self, X: [float]) -> [float]:
        Y_predicted = []
        for x in X:
            prediction = self.forward(x)
            Y_predicted.append(1 if prediction > 0.5 else 0)
        return Y_predicted

    def predict_proba(self, X: [float]) -> [float]:
        """Return probability predictions"""
        Y_predicted = []
        for x in X:
            prediction = self.forward(x)
            Y_predicted.append(prediction)
        return Y_predicted

    # Activation function (Sigmoid)
    @staticmethod
    def sigmoid(z):
        # Clip z to avoid overflow
        z = max(min(z, 500), -500)
        return 1 / (1 + math.exp(-z))

    # Derivative of the Sigmoid function
    @staticmethod
    def sigmoid_derivative(output):
        # Derivative resolution: https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
        return output * (1 - output)

    # Loss function (Binary Cross Entropy)
    @staticmethod
    def compute_loss(predicted, target):
        epsilon = 1e-15  # Small constant to avoid log(0)
        predicted = max(min(predicted, 1 - epsilon), epsilon)  # Clip values
        return -(target * math.log(predicted) + (1 - target) * math.log(1 - predicted))

    # Training function for the artificial neuron
    def train(self, training_data, learning_rate=0.005, epochs=10000, lambda_l2=0.001):
        # Find min and max values for normalization
        x_values = [x for x, _ in training_data]
        self.x_min = min(x_values)
        self.x_max = max(x_values)

        for epoch in range(epochs):
            total_loss = 0

            for x, target in training_data:
                # Forward Pass
                predicted_output = self.forward(x)

                # Calculate Loss (including L2 regularization)
                loss = self.compute_loss(predicted_output, target)
                l2_loss = 0.5 * lambda_l2 * (self.weight ** 2)  # L2 regularization term
                total_loss += loss + l2_loss

                # Backward Pass (Calculate gradients)
                error = predicted_output - target
                sigmoid_grad = self.sigmoid_derivative(predicted_output)
                x_norm = self.normalize_input(x)
                gradient = error * sigmoid_grad * x_norm + lambda_l2 * self.weight  # Add L2 regularization gradient
                bias_gradient = error * sigmoid_grad

                # Update Weight and Bias
                self.weight -= learning_rate * gradient
                self.bias -= learning_rate * bias_gradient

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {total_loss:.4f}, Weight: {self.weight:.4f}, Bias: {self.bias:.4f}")
