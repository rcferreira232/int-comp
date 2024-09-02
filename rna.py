import numpy as np

# Função de ativação sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-0.5 * x))

# Inicialização dos pesos
wi1 = np.random.rand(5)
wi2 = np.random.rand(5)

wj1 = np.random.rand(3)
wj2 = np.random.rand(3)

def forward(X, wi1, wi2, wj1, wj2):
    # Cálculo dos valores de entrada
    vi1 = np.dot(X, wi1)
    vi2 = np.dot(X, wi2)

    first_layer = np.array([1, sigmoid(vi1), sigmoid(vi2)])

    output = np.array(
        [
            sigmoid(np.dot(first_layer, wj1)),
            sigmoid(np.dot(first_layer, wj2)),
        ]
    )

    return output, first_layer

def backward(X, y, output, first_layer, wi1, wi2, wj1, wj2, learning_rate):
    a = 0.5

    # Cálculo do erro
    e1 = y[0] - output[0]
    e2 = y[1] - output[1]

    # Cálculo dos gradientes output
    gradient_output1 = e1 * a * output[0] * (1 - output[0])
    gradient_output2 = e2 * a * output[1] * (1 - output[1])

    # Atualização dos pesos output
    delta_wj1 = learning_rate * gradient_output1 * np.array([1, *output])
    wj1 += delta_wj1

    delta_wj2 = learning_rate * gradient_output2 * np.array([1, *output])
    wj2 += delta_wj2

    # Cálculo dos gradientes first layer
    summation1 = np.dot(
        [gradient_output1, gradient_output2], [wj1[1], wj2[1]]
    )
    summation2 = np.dot(
        [gradient_output1, gradient_output2], [wj1[2], wj2[2]]
    )

    gradient_fl1 = a * first_layer[1] * (1 - first_layer[1]) * summation1
    gradient_fl2 = a * first_layer[2] * (1 - first_layer[2]) * summation2

    # Atualização dos pesos first layer
    delta_wi1 = learning_rate * gradient_fl1 * np.array(X)
    wi1 += delta_wi1

    delta_wi2 = learning_rate * gradient_fl2 * np.array(X)
    wi2 += delta_wi2

    return wi1, wi2, wj1, wj2

def train(X, y, wi1, wi2, wj1, wj2, epochs, learning_rate):
    for epoch in range(epochs):
        for i in range(len(X)):
            output, first_layer = forward(X[i], wi1, wi2, wj1, wj2)
            wi1, wi2, wj1, wj2 = backward(X[i], y[i], output, first_layer, wi1, wi2, wj1, wj2, learning_rate)
    return wi1, wi2, wj1, wj2

def predict(X, wi1, wi2, wj1, wj2):
    output, _ = forward(X, wi1, wi2, wj1, wj2)
    if output[0] > output[1]:
        return "gripe"
    elif output[0] < output[1]:
        return "dengue"
    else:
        return "doença não identificada"

# Exemplo de uso
X = np.array([[1, 1, 0, 1, 0], [1, 0, 1, 0, 1], [1, 0, 0, 1, 0], [1, 0, 0, 0, 1]])
y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])

# Treinamento do modelo
wi1, wi2, wj1, wj2 = train(X, y, wi1, wi2, wj1, wj2, epochs=10000, learning_rate=0.3)

# Predição
print(predict([1, 1, 0, 0, 0], wi1, wi2, wj1, wj2))