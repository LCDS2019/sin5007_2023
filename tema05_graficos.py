
import matplotlib.pyplot as plt
import numpy as np

Todas_as_características = [0.9999, 0.9998, 0.9998, 1.0000]
PCA = [0.9877, 0.9771, 0.9780, 0.9975]
Relief = [0.8666, 0.8266, 0.8697, 0.9401]
RUS = [1.0000, 1.0000, 1.0000, 1.0000]
ROS = [0.9999, 0.9999, 0.9999, 1.0000]

Todas_dp = [0.0002, 0.0000, 0.0005, 0.0004]
PCA_dp = [0.0030, 0.0056, 0.0062, 0.0016]
Relief_dp = [0.2504, 0.2371, 0.2698, 0.0027]
RUS_dp = [0.0000, 0.0000, 0.0000, 0.0000]
ROS_dp = [0.0001, 0.0001, 0.0003, 0.0000]

nomes_das_barras = ['F1-score','Acurácia','Revocação','Precisão']

import matplotlib.pyplot as plt
import numpy as np

# Valores médios para as métricas
nomes_das_barras = ['F1-score', 'Acurácia', 'Revocação', 'Precisão']
Todas_as_características = [0.9999, 0.9998, 0.9998, 1.0000]
PCA = [0.9877, 0.9771, 0.9780, 0.9975]
Relief = [0.8666, 0.8266, 0.8697, 0.9401]
RUS = [1.0000, 1.0000, 1.0000, 1.0000]
ROS = [0.9999, 0.9999, 0.9999, 1.0000]

# Desvios padrão para as métricas
Todas_dp = [0.0002, 0.0000, 0.0005, 0.0004]
PCA_dp = [0.0030, 0.0056, 0.0062, 0.0016]
Relief_dp = [0.2504, 0.2371, 0.2698, 0.0027]
RUS_dp = [0.0000, 0.0000, 0.0000, 0.0000]
ROS_dp = [0.0001, 0.0001, 0.0003, 0.0000]

# Defina a largura das barras
largura_barra = 0.15

# Crie um array com a posição das barras no eixo x
x = np.arange(len(nomes_das_barras))

# Crie o gráfico de barras para cada conjunto de dados
plt.bar(x - largura_barra*2, Todas_as_características, largura_barra, label='Todas as Características', alpha=0.7, yerr=Todas_dp, capsize=5)
plt.bar(x - largura_barra, PCA, largura_barra, label='PCA', alpha=0.7, yerr=PCA_dp, capsize=5)
plt.bar(x, Relief, largura_barra, label='Relief', alpha=0.7, yerr=Relief_dp, capsize=5)
plt.bar(x + largura_barra, RUS, largura_barra, label='RUS', alpha=0.7, yerr=RUS_dp, capsize=5)
plt.bar(x + largura_barra*2, ROS, largura_barra, label='ROS', alpha=0.7, yerr=ROS_dp, capsize=5)

# Configure os rótulos do eixo x e o título
plt.xlabel('Medidas')
plt.ylabel('Valores Médios')
plt.title('Valores Médios e Desvios Padrão por Medidas e Técnica de Seleção de Atributos')
plt.xticks(x, nomes_das_barras)

# Mova a legenda para a parte inferior
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)

# Exiba o gráfico
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Valores médios para as métricas
nomes_das_barras = ['F1-score', 'Acurácia', 'Revocação', 'Precisão']
Todas_as_características = [0.9999, 0.9998, 0.9998, 1.0000]
PCA = [0.9872, 0.9762, 0.9762, 0.9984]
Relief = [0.9456, 0.8975, 0.9480, 0.9432]
RUS = [1.0000, 1.0000, 1.0000, 1.0000]
ROS = [1.0000, 1.0000, 1.0000, 1.0000]

# Defina a largura das barras
largura_barra = 0.15

# Crie um array com a posição das barras no eixo x
x = np.arange(len(nomes_das_barras))

# Função para adicionar os valores no topo de cada barra formatados com duas casas decimais
def add_values(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                 '{:.2f}'.format(height), ha='center', va='bottom')

# Crie o gráfico de barras para cada conjunto de dados
bar1 = plt.bar(x - largura_barra*2, Todas_as_características, largura_barra, label='Todas as Características', alpha=0.7)
bar2 = plt.bar(x - largura_barra, PCA, largura_barra, label='PCA', alpha=0.7)
bar3 = plt.bar(x, Relief, largura_barra, label='Relief', alpha=0.7)
bar4 = plt.bar(x + largura_barra, RUS, largura_barra, label='RUS', alpha=0.7)
bar5 = plt.bar(x + largura_barra*2, ROS, largura_barra, label='ROS', alpha=0.7)

add_values(bar1)
add_values(bar2)
add_values(bar3)
add_values(bar4)
add_values(bar5)

# Configure os rótulos do eixo x e o título
plt.xlabel('Medidas')
plt.ylabel('Valores Médios')
plt.title('Valores Médios por Medidas e Técnica de Seleção de Atributos')
plt.xticks(x, nomes_das_barras)

# Mova a legenda para a parte inferior
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)
plt.ylim(0, 1.1)

# Exiba o gráfico
plt.tight_layout()
plt.show()
