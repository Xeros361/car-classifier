import random
import pandas as pd
from train import train
from data import get_must_results, load_data, normalization, save_min_max_values
from metrics import accuracy, mse
from model import forward_pass
from user_input import get_input, forward_pass_input

og_df = load_data("cars.csv")
cp_df = og_df.copy()
og_test_df = load_data("test_cars.csv")
cp_test_df = og_test_df.copy()

min_values = {}
max_values = {}
save_min_max_values(cp_df, min_values, max_values)
normalization(cp_df, min_values, max_values)
normalization(cp_test_df, min_values, max_values)

must_results = get_must_results(og_df)
results = []
test_must_results = get_must_results(og_test_df)
test_results = []

hidden1_before_relu = []
hidden1_after_relu = []
hidden2_before_relu = []
hidden2_after_relu = []

weights_input = [random.uniform(-0.5, 0.5) for _ in range(16)]
weights_middle = [random.uniform(-0.5, 0.5) for _ in range(16)]
weights_output = [random.uniform(-0.5, 0.5) for _ in range(4)]
bias_hidden1 = [random.uniform(-0.5, 0.5) for _ in range(4)]
bias_hidden2 = [random.uniform(-0.5, 0.5) for _ in range(4)]
bias_output = random.uniform(-0.5, 0.5)

learning_rate = 0.005
epochs = 250

if __name__ == "__main__":
    print("It can take a while... (cause it's not optimized)\n")
    for epoch in range(epochs):
        results = []
        hidden1_before_relu.clear()
        hidden1_after_relu.clear()
        hidden2_before_relu.clear()
        hidden2_after_relu.clear()

        for i in range(len(cp_df)):
            prediction = forward_pass(i, weights_input + weights_middle + weights_output, cp_df, bias_hidden1,
                                      bias_hidden2, bias_output, hidden1_before_relu, hidden1_after_relu,
                                      hidden2_before_relu, hidden2_after_relu)
            results.append(prediction)

        weights_input, weights_middle, weights_output, bias_hidden1, bias_hidden2, bias_output = train(hidden1_before_relu,
                        hidden1_after_relu, hidden2_before_relu, hidden2_after_relu, weights_input, weights_middle,
                        weights_output, bias_hidden1, bias_hidden2, bias_output, results, must_results,
                        learning_rate, cp_df)

        if epoch % 25 == 0:
            print(f"MSE: {mse(must_results, results)}")

    print(f"Final MSE: {mse(must_results, results)}")
    print(f"Final training accuracy: {accuracy(results, must_results)}%")

    for i in range(len(cp_test_df)):
        prediction = forward_pass(i, weights_input + weights_middle + weights_output, cp_test_df, bias_hidden1,
                                      bias_hidden2, bias_output, hidden1_before_relu, hidden1_after_relu,
                                      hidden2_before_relu, hidden2_after_relu)
        test_results.append(prediction)

    print(f"Test accuracy: {accuracy(test_results, test_must_results)}%")

    while True:
        choice = input("Do you want to classify another car? (y/n): ").strip().lower()
        if choice == "y":
            user_input = get_input(min_values, max_values)
            probability = forward_pass_input(user_input, weights_input + weights_middle + weights_output, bias_hidden1,
                                             bias_hidden2, bias_output)
            if probability >= 0.5:
                print("This car is sport")
            else:
                print("This car is NOT sport")
            print(f"Probability that this car is sport: {probability:.4f}")
        else:
            break
