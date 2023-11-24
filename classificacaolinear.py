# Classificação Linear Mulheres modelo 2

# importações
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pickle

# Dados
anos = np.arange(1, 13).reshape(-1, 1)  # Representando os anos de 1 a 12
data = {
    'Homens': [27.4, 28.8, 28.7, 29.0, 28.5, 28.2, 27.2, 26.9, 26.5, 26.1, 21.0, 24.1],
    'Mulheres': [16.1, 16.4, 15.9, 15.1, 14.0, 12.9, 12.0, 11.2, 10.9, 10.7, 8.2, 9.3],
    'Menores de 5 anos': [3.7, 3.4, 3.5, 3.0, 2.7, 2.4, 2.3, 2.1, 1.8, 1.8, 1.2, 1.2],
    'Entre 15 e 24 anos': [11.5, 12.8, 14.0, 14.6, 14.8, 15.2, 14.6, 15.0, 14.4, 14.5, 11.2, 13.3]
}

# Convertendo a variável de resposta em classes (binarizando)
classes = (data[variavel_demografica] > np.mean(data[variavel_demografica])).astype(int)

# Criação do modelo de classificação linear
modelo = LogisticRegression()
modelo.fit(anos, classes)

# Previsão para o próximo ano
proximo_ano = np.array([[13]])
previsao = modelo.predict(proximo_ano)

# Mapeando a previsão para uma mensagem explicativa
mensagem = "Menor ou igual à média" if previsao[0] == 0 else "Maior do que a média"

# Visualização dos resultados
plt.scatter(anos, data[variavel_demografica], color='blue')
plt.plot(anos, modelo.predict(anos), color='red')
plt.scatter(proximo_ano, previsao, color='green', marker='x', s=100)
plt.title(f'Classificação Linear para {variavel_demografica}')
plt.xlabel('Ano')
plt.ylabel('Porcentagem')
plt.show()

print(f"Previsão para o próximo ano: {previsao[0]:.0f} ({mensagem})")