import numpy as np
from sklearn.neural_network import MLPClassifier

X_train = np.array([
    [1, 1, 0, 1, 0],  
    [1, 0, 1, 0, 1],  
    [1, 0, 0, 1, 0],  
    [1, 0, 0, 0, 1]   
])

y_train = np.array([
    [1, 0],  # Gripe
    [0, 1],  # Dengue
    [1, 0],  # Gripe
    [0, 1]   # Dengue
])

# hidden_layer_sizes: O i-ésimo elemento representa o número de neurônios na i-ésima camada oculta.
    # (2,) -> 1 camada oculta com 2 neurônios
# activation: Função de ativação para a camada oculta.
    # "logistic": a função lógica sigmoid, retorna f(x) = 1 / (1 + exp(-x)).
# solver: O otimizador de pesos.
    # "adam": a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
# max_iter: Número máximo de iterações.
    # 1000: 1000 iterações


mlp = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000, activation='logistic', solver='adam')
mlp.fit(X_train, y_train)

def get_symptoms():
    symptoms = [1]  # Começa com o bias, sempre 1

    # Lista de perguntas correspondentes a cada sintoma
    questions = [
        "Você tem tosse? (s/n): ",
        "Você tem manchas na pele? (s/n): ",
        "Você tem coriza? (s/n): ",
        "Você tem baixa contagem de plaquetas? (s/n): "
    ]

    for question in questions:
        answer = input(question).strip().lower()
        if answer == "s":
            symptoms.append(1)
        elif answer == "n":
            symptoms.append(0)
        else:
            print("Resposta inválida. Responda com 's' ou 'n'.")
            return get_symptoms()
        
    return np.array([symptoms])

# Coletando os sintomas do usuário
X_test = get_symptoms()

# Fazendo a previsão
prediction = mlp.predict(X_test)
predict_proba = mlp.predict_proba(X_test)
score = mlp.score(X_train, y_train)


# Interpretando o resultado
if prediction[0][0] == 1:
    print("O diagnóstico sugere Gripe.")
    print("Probabilidade de ser Gripe: {:.2f}%".format(predict_proba[0][0] * 100))
elif prediction[0][1] == 1:
    print("O diagnóstico sugere Dengue.")
    print("Probabilidade de ser Dengue: {:.2f}%".format(predict_proba[0][1] * 100))
else:
    print("Diagnóstico inconclusivo.")

# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html