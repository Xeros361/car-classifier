import math

def forward_pass(n, wghts, df, bias_hidden1, bias_hidden2, bias_output, hidden1_before_relu, hidden1_after_relu,
                 hidden2_before_relu, hidden2_after_relu):
    neurons_hidden_1 = [0, 0, 0, 0]
    neurons_hidden_2 = [0, 0, 0, 0]
    row = df.iloc[n].drop("is_sport")
    w = 0
    s = 0

    for val in row:
        for i in range(4):
            neurons_hidden_1[i] += val * wghts[w]
            w += 1
    for i in range(4):
        neurons_hidden_1[i] += bias_hidden1[i]

    hidden1_before_relu.append(neurons_hidden_1)
    relu(neurons_hidden_1)
    hidden1_after_relu.append(neurons_hidden_1)

    for i in range(4):
        for j in range(4):
            neurons_hidden_2[j] += neurons_hidden_1[i] * wghts[w]
            w += 1
    for i in range(4):
        neurons_hidden_2[i] += bias_hidden2[i]

    hidden2_before_relu.append(neurons_hidden_2)
    relu(neurons_hidden_2)
    hidden2_after_relu.append(neurons_hidden_2)

    for i in range(4):
        s += neurons_hidden_2[i] * wghts[w]
        w += 1
    s += bias_output

    return sigmoid(s)

def relu(n_h):
    for i in range(len(n_h)):
        if n_h[i] <= 0:
            n_h[i] = 0

def sigmoid(x):
    return 1 / (1 + math.exp(-x))