import random

class NeuralNetwork:
    def __init__(self):
        # Random initialize weight
        self.weight = random.random()
        self.bias = random.random()

    def forward(self, _input: float | int):
        # Calculate the weighted sum
        output = self.weight * _input + self.bias
        return output

    def predict(self, X: [float]) -> [float]:
        Y_predicted = []
        for x in X:
            prediction = self.forward(x)
            Y_predicted.append(prediction)
        return Y_predicted

    @staticmethod
    def compute_loss(predicted_output: float, target: float) -> float:
        # In this example we used Mean Square Error (MSE)
        return (predicted_output - target) ** 2

    def compute_total_loss(self, targets: [float], predicted_outputs: [float]) -> float:
        # In this example we used Mean Square Error (MSE)
        total_loss = 0
        for target, predicted_output in zip(targets, predicted_outputs):
            total_loss += (predicted_output - target) ** 2

        return total_loss

    def compute_loss_derivative(self, predicted_output: float, target: float) -> float:
        return (predicted_output - target) * 2

    def train(self, training_sample: [float], learning_rate=0.01, epochs=1000):
        for epoch in range(1, epochs + 1):
            total_loss = 0

            for _input, target in training_sample:
                # Forward Pass
                predicted_output = self.forward(_input)
                # print(_input, target, predicted_output)

                # Calculate Loss
                loss = self.compute_loss(predicted_output, target)
                total_loss += loss

                # Backward Pass (Calculate gradients)
                loss_derivative_value = self.compute_loss_derivative(predicted_output, target)
                gradient = loss_derivative_value * _input
                bias_gradient = loss_derivative_value

                # Update Weights and Bias
                self.weight -= learning_rate * gradient
                self.bias -= learning_rate * bias_gradient

            # Print loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {total_loss:.4f}, Weight: {self.weight:.4f}, Bias: {self.bias:.4f}")


# Use case
if __name__ == "__main__":
    # Example input (X) and outpout (y)
    X = [1, 2, 3, 4]
    y = [2, 4, 6, 8]  # We expect that the neuron learns to multiply by 2

    # Initialize and train the neural net
    neuron = NeuralNetwork()
    population = [*zip(X, y)]
    training_sample = [*zip(X, y)]
    learning_rate = 0.01
    epochs = 1000
    neuron.train(training_sample, learning_rate, epochs)

    print("Predictions after training:")
    for input_data in X:
        print(f"Input: {input_data}, Prediction: {neuron.forward(input_data)}")
