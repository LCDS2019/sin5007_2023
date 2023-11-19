import pandas as pd

from plot import plot_heatmap, plot_table, plot_pair

seed_value = 45
class_name = 'TP_REDE_DEPARA'

df = pd.read_csv('tema01_cursos_2021_ti.csv', index_col=0, sep='|')

columns = [
    class_name,
    'NO_REGIAO',
    'NO_UF',
    'IN_CAPITAL_DEPARA',
    'NO_CURSO_DEPARA',
    'TP_GRAU_ACADEMICO_DEPARA',
    'TP_MODALIDADE_ENSINO_DEPARA',
    'TP_NIVEL_ACADEMICO_DEPARA',
    'TP_DIMENSAO_DEPARA',
    'TP_ORGANIZACAO_ACADEMICA_DEPARA',
    # 'TP_CATEGORIA_ADMINISTRATIVA_DEPARA',
    'QT_VG_TOTAL',
    'QT_INSCRITO_TOTAL',
    'QT_ING',
    'QT_MAT',
    'QT_CONC'
]

df = df[columns]

numeric = [col for col in df.columns if col.startswith('QT_')]
categorical = [col for col in df.columns if not col.startswith('QT_')]

df[numeric] = df[numeric].astype(int)
df[categorical] = df[categorical].astype(str)

# pd.set_option('display.max_columns', None)  # This will display all columns
# pd.set_option('display.expand_frame_repr', False)  # This prevents the DataFrame from being split across multiple lines
# print(df.head)
# plot_table(df.describe().round(3), 'Cursos de TI -- descrição')
# plot_heatmap(df[numeric].corr(), 'Cursos  de TI -- matriz de correlação')
# plot_pair(df.describe().round(3), 'Cursos de TI -- pair plot')

X = df.drop(class_name, axis=1)
y = df[class_name]
