import pandas as pd

seed_value = 45
class_name = 'TP_REDE_DEPARA'

df = pd.read_csv('tema01_cursos_2021_ti_sem_gratuito.csv', index_col=0, sep='|')

# df.drop('IN_GRATUITO_DEPARA', axis=1).to_csv('tema01_cursos_2021_ti_sem_gratuito.csv', sep='|')

X = df.drop(class_name, axis=1)
y = df[class_name]
