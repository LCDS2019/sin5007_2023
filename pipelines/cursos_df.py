import pandas as pd

seed_value = 45
class_name = 'TP_REDE_DEPARA'

df = pd.read_csv('tema01_cursos_2021_ti.csv', index_col=0, sep='|')
X = df.drop(class_name, axis=1)
y = df[class_name]
