import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cursos_ies_2021 = pd.read_csv('arquivos/dados/tema01_cursos_2021.csv',sep='|')
print(cursos_ies_2021.head(10))
cursos_ies_2021 = cursos_ies_2021.fillna(int(0))

print(cursos_ies_2021.head(10))
colunas_selecionadas = ['QT_VG_TOTAL','QT_INSCRITO_TOTAL','QT_ING','QT_MAT','QT_CONC']
resumo_colunas = cursos_ies_2021[colunas_selecionadas].info()
print(resumo_colunas)

resumo_colunas = cursos_ies_2021[colunas_selecionadas].describe()
print(resumo_colunas)

#resumo_colunas.to_csv('arquivos/dados/resumo_colunas.csv')

########################################################################
print(f' filtro de cursos '.center(80 ,'-'))

cursos_ies_2021_filtro = cursos_ies_2021[
(cursos_ies_2021['NO_CURSO'] == 'Abi - Ciência Da Computação') |
(cursos_ies_2021['NO_CURSO'] == 'Administração Em Sistemas E Serviços De Saúde') |
(cursos_ies_2021['NO_CURSO'] == 'Agrocomputação') |
(cursos_ies_2021['NO_CURSO'] == 'Análise De Infraestrutura De Redes E Sistemas Computacionais') |
(cursos_ies_2021['NO_CURSO'] == 'Análise De Sistemas') |
(cursos_ies_2021['NO_CURSO'] == 'Análise E Desenvolvimento De Sistemas') |
(cursos_ies_2021['NO_CURSO'] == 'Ciência Da Computação') |
(cursos_ies_2021['NO_CURSO'] == 'Ciências Da Computação') |
(cursos_ies_2021['NO_CURSO'] == 'Ciências De Computação') |
(cursos_ies_2021['NO_CURSO'] == 'Computação') |
(cursos_ies_2021['NO_CURSO'] == 'Computação E Informática') |
(cursos_ies_2021['NO_CURSO'] == 'Computação E Robótica Educativa') |
(cursos_ies_2021['NO_CURSO'] == 'Computação Em Nuvem') |
(cursos_ies_2021['NO_CURSO'] == 'Computação Gráfica') |
(cursos_ies_2021['NO_CURSO'] == 'Desenvolvimento De Sistemas') |
(cursos_ies_2021['NO_CURSO'] == 'Engenharia Da Computação') |
(cursos_ies_2021['NO_CURSO'] == 'Engenharia De Automação E Sistemas') |
(cursos_ies_2021['NO_CURSO'] == 'Engenharia De Computação') |
(cursos_ies_2021['NO_CURSO'] == 'Engenharia De Computação - Ênfase Sistemas Corporativos') |
(cursos_ies_2021['NO_CURSO'] == 'Engenharia De Computação E Informação') |
(cursos_ies_2021['NO_CURSO'] == 'Engenharia De Produção E Sistemas') |
(cursos_ies_2021['NO_CURSO'] == 'Engenharia De Sistemas') |
(cursos_ies_2021['NO_CURSO'] == 'Engenharia De Sistemas Ciber Físicos') |
(cursos_ies_2021['NO_CURSO'] == 'Engenharia Elétrica  - Ênfase Em Computação') |
(cursos_ies_2021['NO_CURSO'] == 'Engenharia Elétrica - Ênfase Em Eletrônica E Sistemas Computacionais') |
(cursos_ies_2021['NO_CURSO'] == 'Engenharia Eletrônica E De Computação') |
(cursos_ies_2021['NO_CURSO'] == 'Interdisciplinar Em Matemática E Computação E Suas Tecnologias') |
(cursos_ies_2021['NO_CURSO'] == 'Internet Das Coisas E Computação Em Nuvem') |
(cursos_ies_2021['NO_CURSO'] == 'Matemática Aplicada Com Habilitação Em Sistemas E Controle') |
(cursos_ies_2021['NO_CURSO'] == 'Matemática Aplicada E Computação Científica') |
(cursos_ies_2021['NO_CURSO'] == 'Matemática Aplicada E Computacional Com Habilitação Em Sistemas E Controle') |
(cursos_ies_2021['NO_CURSO'] == 'Sistemas De Computação') |
(cursos_ies_2021['NO_CURSO'] == 'Sistemas De Informação') |
(cursos_ies_2021['NO_CURSO'] == 'Sistemas Para Internet')]

########################################################################
print(f' depara de cursos '.center(80 ,'-'))

depara = {
'Abi - Ciência Da Computação':'Ciências da Computação',
'Administração Em Sistemas E Serviços De Saúde':'Análise e Desenvolvimento de Sistemas',
'Agrocomputação':'Agrocomputação',
'Análise De Infraestrutura De Redes E Sistemas Computacionais':'Análise e Desenvolvimento de Sistemas',
'Análise De Sistemas':'Análise e Desenvolvimento de Sistemas',
'Análise E Desenvolvimento De Sistemas':'Análise e Desenvolvimento de Sistemas',
'Ciência Da Computação':'Ciências da Computação',
'Ciências Da Computação':'Ciências da Computação',
'Ciências De Computação':'Ciências da Computação',
'Computação':'Ciências da Computação',
'Computação E Informática':'Ciências da Computação',
'Computação E Robótica Educativa':'Ciências da Computação',
'Computação Em Nuvem':'Ciências da Computação',
'Computação Gráfica':'Ciências da Computação',
'Desenvolvimento De Sistemas':'Análise e Desenvolvimento de Sistemas',
'Engenharia Da Computação':'Engenharia da Computação',
'Engenharia De Automação E Sistemas':'Engenharia de Sistemas',
'Engenharia De Computação':'Engenharia da Computação',
'Engenharia De Computação - Ênfase Sistemas Corporativos':'Engenharia da Computação',
'Engenharia De Computação E Informação':'Engenharia da Computação',
'Engenharia De Produção E Sistemas':'Engenharia de Sistemas',
'Engenharia De Sistemas':'Engenharia de Sistemas',
'Engenharia De Sistemas Ciber Físicos':'Engenharia de Sistemas',
'Engenharia Elétrica  - Ênfase Em Computação':'Engenharia Elétrica  - Ênfase Em Computação',
'Engenharia Elétrica - Ênfase Em Eletrônica E Sistemas Computacionais':'Engenharia Elétrica  - Ênfase Em Computação',
'Engenharia Eletrônica E De Computação':'Engenharia Elétrica  - Ênfase Em Computação',
'Interdisciplinar Em Matemática E Computação E Suas Tecnologias':'Matemática Aplicada e Computação Científica',
'Internet Das Coisas E Computação Em Nuvem':'Ciências da Computação',
'Matemática Aplicada Com Habilitação Em Sistemas E Controle':'Matemática Aplicada e Computação Científica',
'Matemática Aplicada E Computação Científica':'Matemática Aplicada e Computação Científica',
'Matemática Aplicada E Computacional Com Habilitação Em Sistemas E Controle':'Matemática Aplicada e Computação Científica',
'Sistemas De Computação':'Sistemas de Computação',
'Sistemas De Informação':'Sistemas de Informação',
'Sistemas Para Internet':'Análise e Desenvolvimento de Sistemas'
}

depara = {
'Abi - Ciência Da Computação':'CComputação',
'Administração Em Sistemas E Serviços De Saúde':'ADS',
'Agrocomputação':'Agrocomputação',
'Análise De Infraestrutura De Redes E Sistemas Computacionais':'ADS',
'Análise De Sistemas':'ADS',
'Análise E Desenvolvimento De Sistemas':'ADS',
'Ciência Da Computação':'CComputação',
'Ciências Da Computação':'CComputação',
'Ciências De Computação':'CComputação',
'Computação':'CComputação',
'Computação E Informática':'CComputação',
'Computação E Robótica Educativa':'CComputação',
'Computação Em Nuvem':'CComputação',
'Computação Gráfica':'CComputação',
'Desenvolvimento De Sistemas':'ADS',
'Engenharia Da Computação':'Eng. Computação',
'Engenharia De Automação E Sistemas':'Eng. Computação',
'Engenharia De Computação':'Eng. Computação',
'Engenharia De Computação - Ênfase Sistemas Corporativos':'Eng. Computação',
'Engenharia De Computação E Informação':'Eng. Computação',
'Engenharia De Produção E Sistemas':'Eng. de Sistemas',
'Engenharia De Sistemas':'Eng. de Sistemas',
'Engenharia De Sistemas Ciber Físicos':'Eng. de Sistemas',
'Engenharia Elétrica  - Ênfase Em Computação':'Eng. El. Ênfase Em Computação',
'Engenharia Elétrica - Ênfase Em Eletrônica E Sistemas Computacionais':'Eng. El. Ênfase Em Computação',
'Engenharia Eletrônica E De Computação':'Eng. El. Ênfase Em Computação',
'Interdisciplinar Em Matemática E Computação E Suas Tecnologias':'Mat. e CCientífica',
'Internet Das Coisas E Computação Em Nuvem':'CComputação',
'Matemática Aplicada Com Habilitação Em Sistemas E Controle':'Mat. e CCientífica',
'Matemática Aplicada E Computação Científica':'Mat. e CCientífica',
'Matemática Aplicada E Computacional Com Habilitação Em Sistemas E Controle':'Mat. e CCientífica',
'Sistemas De Computação':'SComputação',
'Sistemas De Informação':'SInformação',
'Sistemas Para Internet':'ADS'
}

cursos_ies_2021_filtro['NO_CURSO_DEPARA'] = cursos_ies_2021_filtro['NO_CURSO'].map(depara)

########################################################################
print(f' depara de IN_CAPITAL '.center(80 ,'-'))

depara_IN_CAPITAL = {0:'Não',
                     1:'Sim'}

cursos_ies_2021_filtro['IN_CAPITAL_DEPARA'] = cursos_ies_2021_filtro['IN_CAPITAL'].map(depara_IN_CAPITAL)

########################################################################
print(f' depara de TP_DIMENSAO '.center(80 ,'-'))

depara_TP_DIMENSAO = {1:'Cursos presenciais ofertados no Brasil',
                      2:'Cursos a distância ofertados no Brasil',
                      3:'Cursos a distância com dimensão de dados somente a nível Brasil',
                      4:'Cursos a distância ofertados por instituições brasileiras no exterior'}

cursos_ies_2021_filtro['TP_DIMENSAO_DEPARA'] = cursos_ies_2021_filtro['TP_DIMENSAO'].map(depara_TP_DIMENSAO)

########################################################################
print(f' depara de TP_ORGANIZACAO_ACADEMICA '.center(80 ,'-'))

depara_TP_ORGANIZACAO_ACADEMICA = {1:'Universidade',
                                   2:'Centro Universitário',
                                   3:'Faculdade',
                                   4:'Instituto Federal de Educação, Ciência e Tecnologia',
                                   5:'Centro Federal de Educação Tecnológica'}

cursos_ies_2021_filtro['TP_ORGANIZACAO_ACADEMICA_DEPARA'] = (
    cursos_ies_2021_filtro['TP_ORGANIZACAO_ACADEMICA'].map(depara_TP_ORGANIZACAO_ACADEMICA))

########################################################################
print(f' depara de TP_CATEGORIA_ADMINISTRATIVA '.center(80 ,'-'))

depara_TP_CATEGORIA_ADMINISTRATIVA = {
                                        1:'Pública Federal',
                                        2:'Pública Estadual',
                                        3:'Pública Municipal',
                                        4:'Privada com fins lucrativos',
                                        5:'Privada sem fins lucrativos',
                                        6:'Privada - Particular em sentido estrito',
                                        7:'Especial',
                                        8:'Privada comunitária',
                                        9:'Privada confessional'
                                        }

cursos_ies_2021_filtro['TP_CATEGORIA_ADMINISTRATIVA_DEPARA'] = (
    cursos_ies_2021_filtro['TP_CATEGORIA_ADMINISTRATIVA'].map(depara_TP_CATEGORIA_ADMINISTRATIVA))


########################################################################
print(f' depara de TP_REDE '.center(80 ,'-'))

depara_TP_REDE = {
    1:'Pública',
    2:'Privada'}

cursos_ies_2021_filtro['TP_REDE_DEPARA'] = cursos_ies_2021_filtro['TP_REDE'].map(depara_TP_REDE)

########################################################################
print(f' depara de TP_GRAU_ACADEMICO '.center(80 ,'-'))

depara_TP_GRAU_ACADEMICO = {
    1:'Bacharelado',
    2:'Licenciatura',
    3:'Tecnológico',
    4:'Bacharelado e Licenciatura'}

cursos_ies_2021_filtro['TP_GRAU_ACADEMICO_DEPARA'] = (
    cursos_ies_2021_filtro['TP_GRAU_ACADEMICO'].map(depara_TP_GRAU_ACADEMICO))


########################################################################
print(f' depara de IN_GRATUITO '.center(80 ,'-'))

depara_IN_GRATUITO = {
    0:'Não',
    1:'Sim'
}

cursos_ies_2021_filtro['IN_GRATUITO_DEPARA'] = (
    cursos_ies_2021_filtro['IN_GRATUITO'].map(depara_IN_GRATUITO))

########################################################################
print(f' depara de TP_MODALIDADE_ENSINO '.center(80 ,'-'))

depara_TP_MODALIDADE_ENSINO = {
    1:'Presencial',
    2:'Curso a distância'
}

cursos_ies_2021_filtro['TP_MODALIDADE_ENSINO_DEPARA'] = (
    cursos_ies_2021_filtro['TP_MODALIDADE_ENSINO'].map(depara_TP_MODALIDADE_ENSINO))

########################################################################
print(f' depara de TP_NIVEL_ACADEMICO '.center(80 ,'-'))

depara_TP_NIVEL_ACADEMICO = {
    1:'Graduação',
    2:'Sequencial de Formação Específica'
}

cursos_ies_2021_filtro['TP_NIVEL_ACADEMICO_DEPARA'] = (
    cursos_ies_2021_filtro['TP_NIVEL_ACADEMICO'].map(depara_TP_NIVEL_ACADEMICO))

print(cursos_ies_2021_filtro)

cursos_ies_2021_filtro_vf= cursos_ies_2021_filtro[[
    'NU_ANO_CENSO',
    'NO_REGIAO',
    'NO_UF',
    'SG_UF',
    'NO_MUNICIPIO',
    'CO_MUNICIPIO',
    'IN_CAPITAL_DEPARA',
    'CO_IES',
    'SG_IES',
    'NO_IES',
    'CO_MANTENEDORA',
    'NO_MANTENEDORA',
    'NO_CURSO',
    'NO_CURSO_DEPARA',
    'CO_CURSO',
    'TP_GRAU_ACADEMICO_DEPARA',
    'IN_GRATUITO_DEPARA',
    'TP_MODALIDADE_ENSINO_DEPARA',
    'TP_NIVEL_ACADEMICO_DEPARA',
    'TP_DIMENSAO_DEPARA',
    'TP_ORGANIZACAO_ACADEMICA_DEPARA',
    'TP_CATEGORIA_ADMINISTRATIVA_DEPARA',
    'TP_REDE_DEPARA',
    'QT_VG_TOTAL',
    'QT_INSCRITO_TOTAL',
    'QT_ING',
    'QT_MAT',
    'QT_CONC',
]]

########################################################################
print(f' Ajuste de tipos de dados '.center(80 ,'#'))
print('')
cursos_ies_2021_filtro_vf['NU_ANO_CENSO'] = cursos_ies_2021_filtro_vf['NU_ANO_CENSO'].astype(str)
cursos_ies_2021_filtro_vf['CO_MUNICIPIO'] = cursos_ies_2021_filtro_vf['CO_MUNICIPIO'].astype(str)
cursos_ies_2021_filtro_vf['CO_IES'] = cursos_ies_2021_filtro_vf['CO_IES'].astype(str)
cursos_ies_2021_filtro_vf['CO_MANTENEDORA'] = cursos_ies_2021_filtro_vf['CO_MANTENEDORA'].astype(str)
cursos_ies_2021_filtro_vf['CO_CURSO'] = cursos_ies_2021_filtro_vf['CO_CURSO'].astype(str)

cursos_ies_2021_filtro_vf.to_csv('arquivos/dados/tema01_cursos_2021_ti.csv',sep='|', index=False)










