
import matplotlib.pyplot as plt
import numpy as np

# Valores médios para as métricas
nomes_das_barras = ['F1-score', 'Acurácia', 'Revocação', 'Precisão']
Total = [0.9998,0.9997,0.9997,0.9999]
PCA = [0.9958,0.9921, 0.9939, 0.9977]
Relief = [0.9634,0.9296, 0.9876, 0.9404]
RUS = [0.9986,0.9974, 0.9975, 0.9998]
ROS = [0.9998,0.9997, 0.9997, 0.9999]

# Desvios padrão para as métricas
Total_dp = [0.0002,0.0003, 0.0002, 0.0001]
PCA_dp = [0.0010,0.0019, 0.0015, 0.0007]
Relief_dp = [0.0039,0.0073, 0.0101, 0.0040]
RUS_dp = [0.0007,0.0014, 0.0015, 0.0001]
ROS_dp = [0.0002,0.0003, 0.0002, 0.0001]

# Defina a largura das barras
largura_barra = 0.15

# Crie um array com a posição das barras no eixo x
x = np.arange(len(nomes_das_barras))

plt.figure(figsize=(10, 6))

# Crie o gráfico de barras para cada conjunto de dados
plt.bar(x - largura_barra*2, Total, largura_barra, label='Total', alpha=0.7, yerr=Total_dp, capsize=5)
plt.bar(x - largura_barra, PCA, largura_barra, label='PCA', alpha=0.7, yerr=PCA_dp, capsize=5)
plt.bar(x, Relief, largura_barra, label='Relief', alpha=0.7, yerr=Relief_dp, capsize=5)
plt.bar(x + largura_barra, RUS, largura_barra, label='RUS', alpha=0.7, yerr=RUS_dp, capsize=5)
plt.bar(x + largura_barra*2, ROS, largura_barra, label='ROS', alpha=0.7, yerr=ROS_dp, capsize=5)

# Configure os rótulos do eixo x e o título
plt.xlabel('Medidas')
plt.ylabel('Valores Médios')
plt.title('Valores Médios e Desvios Padrão por Medidas e Técnica de Seleção de Atributos - Validação')
plt.xticks(x, nomes_das_barras)

# Mova a legenda para a parte inferior
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)

plt.ylim(0.8, 1.05)

# Exiba o gráfico
plt.tight_layout()
plt.show()

##########################################################################################3

# Valores médios para as métricas
nomes_das_barras = ['F1-score', 'Acurácia', 'Revocação', 'Precisão']
Total = [0.9999,0.9998,0.9999,0.9999]
PCA = [0.9952,0.9911, 0.9934, 0.9971]
Relief = [0.9629,0.9287, 0.9868, 0.9402]
RUS = [0.9985,0.9972, 0.9972, 0.9999]
ROS = [0.9999,0.9998,0.9999,0.9999]

# Desvios padrão para as métricas
Total_dp = [0.0001,0.0001, 0.0001, 0.0001]
PCA_dp = [0.0012,0.0022, 0.0018, 0.0008]
Relief_dp = [0.0048,0.0088, 0.0103,0.0005]
RUS_dp = [0.0006,0.0011, 0.0010, 0.0002]
ROS_dp = [0.0001,0.0001, 0.0001, 0.0001]

# Defina a largura das barras
largura_barra = 0.15

# Crie um array com a posição das barras no eixo x
x = np.arange(len(nomes_das_barras))

plt.figure(figsize=(10, 6))

# Crie o gráfico de barras para cada conjunto de dados
plt.bar(x - largura_barra*2, Total, largura_barra, label='Total', alpha=0.7, yerr=Total_dp, capsize=5)
plt.bar(x - largura_barra, PCA, largura_barra, label='PCA', alpha=0.7, yerr=PCA_dp, capsize=5)
plt.bar(x, Relief, largura_barra, label='Relief', alpha=0.7, yerr=Relief_dp, capsize=5)
plt.bar(x + largura_barra, RUS, largura_barra, label='RUS', alpha=0.7, yerr=RUS_dp, capsize=5)
plt.bar(x + largura_barra*2, ROS, largura_barra, label='ROS', alpha=0.7, yerr=ROS_dp, capsize=5)

# Configure os rótulos do eixo x e o título
plt.xlabel('Medidas')
plt.ylabel('Valores Médios')
plt.title('Valores Médios e Desvios Padrão por Medidas e Técnica de Seleção de Atributos - Teste')
plt.xticks(x, nomes_das_barras)

# Mova a legenda para a parte inferior
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5)

plt.ylim(0.8, 1.05)

# Exiba o gráfico
plt.tight_layout()
plt.show()