from typing import Dict

from pandas import DataFrame

courses: Dict[str, str] = {
    'Agrocomputação': 'Agrocomputação',

    'Administração Em Sistemas E Serviços De Saúde': 'ADS',
    'Análise De Infraestrutura De Redes E Sistemas Computacionais': 'ADS',
    'Análise De Sistemas': 'ADS',
    'Análise E Desenvolvimento De Sistemas': 'ADS',
    'Desenvolvimento De Sistemas': 'ADS',
    'Sistemas Para Internet': 'ADS',

    'Abi - Ciência Da Computação': 'CComputação',
    'Ciência Da Computação': 'CComputação',
    'Ciências Da Computação': 'CComputação',
    'Ciências De Computação': 'CComputação',
    'Computação': 'CComputação',
    'Computação E Informática': 'CComputação',
    'Computação E Robótica Educativa': 'CComputação',
    'Computação Em Nuvem': 'CComputação',
    'Computação Gráfica': 'CComputação',
    'Internet Das Coisas E Computação Em Nuvem': 'CComputação',

    'Engenharia Da Computação': 'Eng. Computação',
    'Engenharia De Automação E Sistemas': 'Eng. Computação',
    'Engenharia De Computação': 'Eng. Computação',
    'Engenharia De Computação - Ênfase Sistemas Corporativos': 'Eng. Computação',
    'Engenharia De Computação E Informação': 'Eng. Computação',

    'Engenharia De Produção E Sistemas': 'Eng. de Sistemas',
    'Engenharia De Sistemas': 'Eng. de Sistemas',
    'Engenharia De Sistemas Ciber Físicos': 'Eng. de Sistemas',

    'Engenharia Elétrica  - Ênfase Em Computação': 'Eng. El. Ênfase Em Computação',
    'Engenharia Elétrica - Ênfase Em Eletrônica E Sistemas Computacionais': 'Eng. El. Ênfase Em Computação',
    'Engenharia Eletrônica E De Computação': 'Eng. El. Ênfase Em Computação',

    'Interdisciplinar Em Matemática E Computação E Suas Tecnologias': 'Mat. e CCientífica',
    'Matemática Aplicada Com Habilitação Em Sistemas E Controle': 'Mat. e CCientífica',
    'Matemática Aplicada E Computação Científica': 'Mat. e CCientífica',
    'Matemática Aplicada E Computacional Com Habilitação Em Sistemas E Controle': 'Mat. e CCientífica',
    'Sistemas De Computação': 'SComputação',
    'Sistemas De Informação': 'SInformação'
}


computer_science_courses = courses.values()


boolean_type: Dict = {0: 'Não', 1: 'Sim'}

offer_type: Dict = {
    1: 'Cursos presenciais ofertados no Brasil',
    2: 'Cursos a distância ofertados no Brasil',
    3: 'Cursos a distância com dimensão de dados somente a nível Brasil',
    4: 'Cursos a distância ofertados por instituições brasileiras no exterior'
}

organization_type: Dict = {
    1: 'Universidade',
    2: 'Centro Universitário',
    3: 'Faculdade',
    4: 'Instituto Federal de Educação, Ciência e Tecnologia',
    5: 'Centro Federal de Educação Tecnológica'
}

administrative_category: Dict = {
    1: 'Pública Federal',
    2: 'Pública Estadual',
    3: 'Pública Municipal',
    4: 'Privada com fins lucrativos',
    5: 'Privada sem fins lucrativos',
    6: 'Privada - Particular em sentido estrito',
    7: 'Especial',
    8: 'Privada comunitária',
    9: 'Privada confessional'
}

network_type: Dict = {
    1: 'Pública',
    2: 'Privada'
}

academic_degree: Dict = {
    1: 'Bacharelado',
    2: 'Licenciatura',
    3: 'Tecnológico',
    4: 'Bacharelado e Licenciatura'
}

teaching_modality: Dict = {
    1: 'Presencial',
    2: 'Curso a distância'
}

not_really_numeric_columns = [
    'NU_ANO_CENSO',
    'CO_MUNICIPIO',
    'CO_IES',
    'CO_MANTENEDORA',
    'CO_CURSO',
    'CO_REGIAO',
    'CO_UF',
    'IN_CAPITAL',
    'TP_DIMENSAO',
    'TP_ORGANIZACAO_ACADEMICA',
    'TP_CATEGORIA_ADMINISTRATIVA',
    'TP_REDE',
    'TP_GRAU_ACADEMICO',
    'IN_GRATUITO',
    'TP_MODALIDADE_ENSINO',
    'TP_NIVEL_ACADEMICO'
]


academic_level: Dict = {
    1: 'Graduação',
    2: 'Sequencial de Formação Específica'
}


dictionaries: Dict[str, Dict] = {
    'NO_CURSO': courses,
    'IN_CAPITAL': boolean_type,
    'TP_DIMENSAO': offer_type,
    'TP_ORGANIZACAO_ACADEMICA': organization_type,
    'TP_CATEGORIA_ADMINISTRATIVA': administrative_category,
    'TP_REDE': network_type,
    'TP_GRAU_ACADEMICO': academic_degree,
    'IN_GRATUITO': boolean_type,
    'TP_MODALIDADE_ENSINO': teaching_modality,
    'TP_NIVEL_ACADEMICO': academic_level
}


def assign_categorical_number_as_str(df: DataFrame):
    for column in not_really_numeric_columns:
        df[column] = df[column].astype(str)


def standardize_column(df: DataFrame, column: str, dict: Dict):
    df[f'{column}_DEPARA'] = df[column].map(dict)


def standardize_columns(df: DataFrame):
    for column in dictionaries:
        standardize_column(df, column, dictionaries[column])

    assign_categorical_number_as_str(df)


def only_computer_science_related(df: DataFrame):
    indices_to_drop = df[~df['NO_CURSO_DEPARA'].isin(computer_science_courses)].index
    df.drop(indices_to_drop, inplace=True)
