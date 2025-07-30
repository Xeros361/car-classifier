from model import relu, sigmoid

def get_input(min_values, max_values):
    speed = float(input("Insert max speed: "))
    wght = float(input("Insert weight: "))
    acc = float(input("Insert acceleration: "))
    seats = int(input("Insert number of seats: "))
    inputs_raw = {"max_speed": speed, "weight": wght, "acceleration": acc, "num_seats": seats}
    inputs = []

    for col in ["max_speed", "weight", "acceleration", "num_seats"]:
        val = inputs_raw[col]
        min_val = min_values[col]
        max_val = max_values[col]
        normalized = (val - min_val) / (max_val - min_val)
        inputs.append(normalized)
    return inputs

def forward_pass_input(input, wghts, bias_hidden1, bias_hidden2, bias_output):
    neurons_hidden_1 = [0, 0, 0, 0]
    neurons_hidden_2 = [0, 0, 0, 0]
    w = 0
    s = 0

    for j in range(4):
        for i in range(4):
            neurons_hidden_1[j] += input[i] * wghts[w]
            w += 1
        neurons_hidden_1[j] += bias_hidden1[j]

    relu(neurons_hidden_1)

    for j in range(4):
        for i in range(4):
            neurons_hidden_2[j] += neurons_hidden_1[i] * wghts[w]
            w += 1
        neurons_hidden_2[j] += bias_hidden2[j]

    relu(neurons_hidden_2)


    for i in range(4):
        s += neurons_hidden_2[i] * wghts[w]
        w += 1
    s += bias_output

    probability = sigmoid(s)
    return probability