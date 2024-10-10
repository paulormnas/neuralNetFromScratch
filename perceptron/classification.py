import random
import math

class NeuralNetwork:
    def __init__(self, num_inputs):
        # Random initialize weight
        self.weights = [random.random() for _ in range(num_inputs)]
        self.bias = random.random()

    def forward(self, _inputs: [float]):
        # Calculate the weighted sum
        z = sum(self.weights[i] * _inputs[i] for i in range(len(_inputs))) + self.bias
        output = self.sigmoid(z)
        return output

    def predict(self, X: [[float]]) -> [float]:
        Y_predicted = []
        for x in X:
            prediction = self.forward(x)
            Y_predicted.append(prediction)
        return Y_predicted

    # Activation function (Sigmoid)
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + math.exp(-z))


    # Derivative of the Sigmoid function
    @staticmethod
    def sigmoid_derivative(output):
        # Derivative resolution: https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
        return output * (1 - output)


    # Loss function (Mean Squared Error)
    @staticmethod
    def compute_loss(predicted, target):
        return (predicted - target) ** 2


    # Derivative of the loss function
    @staticmethod
    def compute_loss_derivative(predicted, target):
        return predicted - target


    # Training function for the artificial neuron
    def train(self, training_data, learning_rate, epochs):
        # Initialize weights and bias
        num_inputs = len(training_data[0][0])

        for epoch in range(epochs):
            total_loss = 0

            for inputs, target in training_data:
                # Forward Pass
                z = sum(self.weights[i] * inputs[i] for i in range(num_inputs)) + self.bias
                predicted_output = self.sigmoid(z)

                # Calculate Loss
                loss = self.compute_loss(predicted_output, target)
                total_loss += loss

                # Backward Pass (Calculate gradients)
                loss_derivative_value = self.compute_loss_derivative(predicted_output, target)
                gradients = [loss_derivative_value * self.sigmoid_derivative(predicted_output) * inputs[i] for i in
                             range(num_inputs)]
                bias_gradient = loss_derivative_value * self.sigmoid_derivative(predicted_output)

                # Update Weights and Bias
                for i in range(num_inputs):
                    self.weights[i] -= learning_rate * gradients[i]
                self.bias -= learning_rate * bias_gradient

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {total_loss:.4f}")



if __name__ == "__main__":
    # Example dataset: AND gate
    # Features and corresponding targets
    x = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    y = [0, 0, 0, 1]
    training_data = [*zip(x, y)]

    # Train the neuron
    neuron = NeuralNetwork(num_inputs=2)
    learning_rate = 0.1
    epochs = 4000
    neuron.train(training_data, learning_rate, epochs)

    # Print the final weights and bias
    print("Final Weights:", neuron.weights)
    print("Final Bias:", neuron.bias)

    # Testing the trained model
    y_predicted = neuron.predict(x)
    predicted_data = [*zip(x, y_predicted, y)]
    for inputs, predicted_output, target in predicted_data:
        print(f"Input: {inputs}, Predicted: {predicted_output:.4f}, Target: {target}")
