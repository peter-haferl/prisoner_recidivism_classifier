import missingno as msno
import numpy as np
import pandas as pd


# load variable dictionaries
variable_names = {'V4': 'case_id',
                  'V8': 'birth_year',
                  'V9': 'sex',
                  'V10': 'race',
                  'V15': 'admission_year',
                  'V24': 'prior_jail_time',
                  'V25': 'prior_prison_time',
                  'V26': 'offense_1',
                  'V32': 'offense_longest_sentence',
                  'V33': 'length_longest_sentence',
                  'V34': 'total_max_sentence_length',
                  'V46': 'year_prison_release',
                  'V48': 'custody_agency_1',
                  'V51': 'release_type',
                  'V56': 'release_supervision_status',
                  'V57': 'age_admission',
                  'V58': 'age_prison_release',
                  'V59': 'age_parole_release',
                  'V62': 'time_served_current_admission',
                  'V67': 'time_served_parole',
                  'V70': 'total_max_sentence_indicator',
                  'V71': 'number_of_offenses_indicator',
                  'V94': 'state'}

variable_categories = {'key': ['case_id'],
                       'datetime': ['birth_year', 'admission_year', 'year_prison_release'],
                       'categorical': ['sex', 'race', 'offense_1', 'offense_longest_sentence', 'custody_agency_1',
                                       'release_type', 'release_supervision_status', 'number_of_offenses_indicator',
                                       'state'],
                       'numeric': ['prior_jail_time', 'prior_prison_time', 'age_admission', 'age_parole_release',
                                   'time_served_current_admission', 'time_served_parole'],
                       'mixed_numeric': ['length_longest_sentence', 'total_max_sentence_length',
                                         'total_max_sentence_indicator']}



def replace_missing(x):
    """replaces dictionary-described missing values with nans"""
    if len(str(x)) == 1 and x in [8, 5, 9]:
        return np.nan

    if len(str(x)) == 2 and x in [88, 95, 98, 99, 97]:
        return np.nan

    if len(str(x)) == 3 and x in [888, 995, 998, 999, 98.0]:
        return np.nan

    if len(str(x)) == 4 and x in [88.8, 99.5, 99.8, 8888, 9995, 9998, 9999]:
        return np.nan

    if len(str(x)) == 5 and x in [888.8, 999.5, 999.8, 9999.9, 88888, 99995, 99998, 99999]:
        return np.nan

    if len(str(x)) == 6 and x in [8888.8, 9999.5, 9999.8]:
        return np.nan

    else:
        return x


def replace_life(x):
    if len(str(x)) == 5 and x in [9999.3, 9999.4, 9999.6, 99993, 99994, 99996]:
        return 29*12


def make_singular_variable_list(data):
    """returns a list of variables with only one value"""
    singular_variables = []
    for col in data.columns:
        if data[col].nunique() == 1:
            singular_variables.append(col)
    return singular_variables


def list_of_absent_data_columns(data, ratio=0.2):
    """returns list of columns whose missing values make up \
    more than a set ratio (default is half) of the data"""
    variables_list = list(data.columns)
    missing_list = []
    for variable in variables_list:
        missing_ratio = sum(data[variable].isna()) / len(data)
        if missing_ratio > ratio:
            missing_list.append(variable)
    return missing_list


def missing_vis(data):
    """Returns 4 graphics: bar chart, distributions, heatmap \
    (of correlation between nulls), and dendogram"""
    msno.bar(data);
    msno.matrix(data);
    msno.heatmap(data);


def clean_target(data):
    """Cleans and labels target (parole) variable"""
    data_clean = data[data['V55'] != 10].copy()
    data_clean['outcome'] = data_clean['V55'].map(lambda x: 1 if x == 1 else 0)
    data_clean.drop(columns='V55', inplace=True)
    data_clean.drop(columns=['V96', 'V97', 'V98', 'V99'], inplace=True)
    return data_clean

def full_clean(data):
    """cleans and transforms raw data to clean"""
    # drop absent/single value columns and values
    data = data.applymap(lambda x: replace_missing(x)).copy()
    absent_columns = list_of_absent_data_columns(data)
    data.drop(columns=absent_columns, inplace=True)
    singular_variables = make_singular_variable_list(data)
    data.drop(columns=singular_variables, inplace=True)
    data = data.dropna()
    data_clean = clean_target(data)
    # create dummy categoricals
    descriptive_columns = list(variable_names.values())
    descriptive_columns.append('outcome')
    data_clean.columns = descriptive_columns
    data_clean = pd.get_dummies(data_clean, columns=variable_categories['categorical'],
                                drop_first=True)
    # drop race and gender columns
    data_clean.drop(columns=['race_2.0', 'race_3.0', 'race_4.0', 'race_6.0', 'sex_2.0'], inplace=True)
    # convert life sentences to average life sentence (348 months)
    data_clean = data_clean.applymap(lambda x: replace_life(x)).copy()
    return data_clean