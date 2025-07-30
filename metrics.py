def mse(real, predicted):
    s = 0
    for i in range(len(real)):
        s += (real[i] - predicted[i]) ** 2
    return s / len(real)

def accuracy(results, must_results):
    correct = 0
    for i in range(len(results)):
        prediction = int(results[i] >= 0.5)
        if must_results[i] == prediction:
            correct += 1
    acr = correct / len(results)
    return round(acr * 100, 2)