def train(hidden1_before_relu, hidden1_after_relu, hidden2_before_relu, hidden2_after_relu, weights_input, weights_middle, weights_output,
          bias_hidden1, bias_hidden2, bias_output, results, must_results, learning_rate, cp_df):

    for i in range(len(cp_df)):
        r = results
        m_r = must_results

        error = r[i] - m_r[i]
        d_sigmoid = r[i] * (1 - r[i])
        d_output = error * d_sigmoid

        for j in range(4):
            gradient = d_output * hidden2_after_relu[i][j]
            weights_output[j] -= learning_rate * gradient
        bias_output -= learning_rate * d_output

        d_hidden2 = [0, 0, 0, 0]
        for j in range(4):
            if hidden2_before_relu[i][j] <= 0:
                d_hidden2[j] = 0
            else:
                d_hidden2[j] = d_output * weights_output[j]
        for j in range(4):
            bias_hidden2[j] -= learning_rate * d_hidden2[j]

        d_hidden1 = [0, 0, 0, 0]
        for j in range(4):
            for h in range(4):
                gradient = d_hidden2[j] * hidden1_after_relu[i][h]
                weights_middle[j * 4 + h] -= learning_rate * gradient
                d_hidden1[h] += d_hidden2[j] * weights_middle[j * 4 + h]
        for j in range(4):
            bias_hidden1[j] -= learning_rate * d_hidden1[j]

        for j in range(4):
            if hidden1_before_relu[i][j] <= 0:
                continue
            for h in range(4):
                data = cp_df.iloc[i].drop("is_sport").values[h]
                gradient = d_hidden1[j] * data
                weights_input[j * 4 + h] -= learning_rate * gradient

    return weights_input, weights_middle, weights_output, bias_hidden1, bias_hidden2, bias_output