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
                                       'state', 'first_offense_type',
                                       'longest_offense_type'],
                       'numeric': ['prior_jail_time', 'prior_prison_time', 'age_admission', 'age_parole_release',
                                   'time_served_current_admission', 'time_served_parole'],
                       'mixed_numeric': ['length_longest_sentence', 'total_max_sentence_length',
                                         'total_max_sentence_indicator']}

state_codes = {
    1: 'AL',
    2: 'AK',
    4: 'AZ',
    5: 'AR',
    6: 'CA',
    8: 'CO',
    19: 'IA',
    21: 'KY',
    22: 'LA',
    24: 'MD',
    26: 'MI',
    29: 'MO',
    31: 'NE',
    32: 'NV',
    34: 'NJ',
    36: 'NY',
    37: 'NC',
    38: 'ND',
    40: 'OK',
    41: 'OR',
    42: 'PA',
    45: 'SC',
    46: 'SD',
    47: 'TN',
    48: 'TX',
    49: 'UT',
    51: 'VA',
    53: 'WA',
    54: 'WV',
    55: 'WI',
    58: 'CYA'
}

offense_codes = {
    'murder': 12,
    'homicide': 14,
    'voluntary manslaughter': 32,
    'kidnapping': 42,
    'rape': 62,
    'sexual assault': 72,
    'lewd act with children': 82,
    'armed robbery': 92,
    'unarmed robbery': 102,
    'forcible sodomy': 112,
    'aggravated assault': 122,
    'simple assault': 132,
    'assualting public officer': 142,
    'blackmail/extortion/intimidation': 152,
    'hit and run': 162,
    'child abuse': 172,
    'violent offences - other': 180,
    'burglary': 192,
    'arson': 202,
    'auto theft': 212,
    'forgery/fraud': 222,
    'grand larceny': 232,
    'petty larceny': 242,
    'larceny - value unknown': 252,
    'embezzlement': 262,
    'stolen property - receiving': 272,
    'stolen property - trafficking': 282,
    'destruction of property': 292,
    'hit/run driving - property damage': 300,
    'unauthorized use of vehicle': 310,
    'trespassing': 322,
    'property offense - other': 335,
    'trafficking - heroin': 342,
    'trafficking - cocaine or crack': 347,
    'trafficking - other controlled': 352,
    'trafficking - marijuana': 362,
    'trafficking - unspecified': 372,
    'possession/use - heroin': 382,
    'possession/use - cocaine or crack': 387,
    'possession/use - other controlled': 392,
    'possession/use - marijuana': 402,
    'possession/use - unspecified': 410,
    'heroin violation - offense unspecified': 420,
    'cocaine or crack violation - offense unspecified': 425,
    'controlled substance- offense unspecified': 430,
    'marijuana violation - offense unspecified': 440,
    'other drug violation - offense unspecified': 450,
    'escape from custody': 462,
    'flight to avoid prosecution': 472,
    'weapon offense': 482,
    'parole violation': 490,
    'probation violation': 500,
    'rioting': 512,
    'habitual offender': 520,
    'contempt of court': 530,
    'offenses against courts, legislature and commissions': 542,
    'traffic offenses - minor': 550,
    'driving while intoxicated': 560,
    'driving under the influence': 565,
    'driving under influence - drugs': 570,
    'family related offenses': 580,
    'drunkenness/vagrancy/disorderly conduct': 590,
    'morals/decency - offense': 602,
    'immigration violations': 610,
    'obstruction - law enforcement': 620,
    'invasion of privacy': 630,
    'commercialized vice': 640,
    'contributing to the delinquency of a minor': 650,
    'liquor law violations': 660,
    'public order offences - other': 672,
    'bribery and conflict of interest': 675,
    'juvenile offenses': 680,
    'felony - unspecified': 692,
    'misdemeanor unspecified': 700,
    'other/unknown': 710,
    'federal - embezzlement': 800,
    'federal - fraud': 810,
    'federal - forgery': 820,
    'federal - counterfeiting': 830,
    'federal - regulatory offenses': 840,
    'federal - tax law': 850,
    'federal - racketeering': 860
}


def offense_bin(series):
    """bins offenses into types of crime based on Bureau of Justice Statistics Offense Codes 
    (http://www.ncrp.info/SiteAssets/Lists/FAQ%20Agencies/EditForm/BJS%20Offense%20Codes.pdf)"""
    copy = []
    for x in series:
        if x <= 180:
            copy.append('violent crime')
        elif x <= 335:
            copy.append('property crime')
        elif x <= 372:
            copy.append('drug trafficking')
        elif x <= 410:
            copy.append('drug possession/use')
        elif x <= 450:
            copy.append('drug crime unspecified')
        elif x <= 542:
            copy.append('noncompliance')
        elif x <= 570:
            copy.append('driving under the influence')
        elif x <= 710:
            copy.append('other')
        else:
            copy.append('federal nonviolent crime')
    return copy


def create_dummy_list(data):
    to_dummy = []
    for x in variable_categories['categorical']:
        if x in list(data.columns):
            to_dummy.append(x)
    return to_dummy


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

    if len(str(x)) == 5 and x in [888.8, 999.5, 999.8, 88888, 99995, 99998, 99999]:
        return np.nan

    if len(str(x)) == 6 and x in [8888.8, 9999.5, 9999.8, 9999.9]:
        return np.nan

    else:
        return x


def replace_life(x):
    Average_life_sentence = 29 * 12
    if x in [99993, 99994, 99996]:
        return Average_life_sentence
    elif x in [9999.3, 9999.4, 9999.6]:
        return Average_life_sentence
    else:
        return x


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


def full_clean():
    """cleans and transforms raw data to clean"""
    # drop absent/single value columns and values
    data = pd.read_csv('./data/da26521-0003.tsv', sep='\t', header=0, low_memory=False)
    data = data.applymap(lambda x: replace_missing(x)).copy()
    absent_columns = list_of_absent_data_columns(data)
    data.drop(columns=absent_columns, inplace=True)
    singular_variables = make_singular_variable_list(data)
    data.drop(columns=singular_variables, inplace=True)
    data = data.dropna()
    data = data.replace({"V94": state_codes})
    data_clean = clean_target(data)
    # create dummy categoricals
    data_clean.rename(index=str, columns=variable_names, inplace=True)
    data_clean = data_clean.loc[data_clean.length_longest_sentence < 99999]
    data_clean = data_clean.loc[data_clean.total_max_sentence_length < 99999]
    data_clean['first_offense_type'] = offense_bin(data_clean.offense_1)
    data_clean['longest_offense_type'] = offense_bin(data_clean.offense_longest_sentence)
    #     print(data_clean.describe(include='all').T)
    to_dummy = create_dummy_list(data_clean)
    data_clean2 = pd.get_dummies(data_clean, columns=to_dummy,
                                 drop_first=True)
    data_clean2['state'] = data_clean['state']
    data_clean2['offense_1'] = data_clean['offense_1']
    data_clean2['first_offense_type'] = data_clean['first_offense_type']
    data_clean2['longest_offense_type'] = data_clean['longest_offense_type']
    data_clean2['offense_longest_sentence'] = data_clean['offense_longest_sentence']
    #     print(list(data_clean2.columns))
    # drop race and gender columns
    data_clean2.drop(columns=['race_2.0', 'race_3.0', 'race_4.0', 'race_6.0',
                              'case_id', 'time_served_parole', 'age_parole_release', 'sex_2.0'], inplace=True)
    # convert life sentences to average life sentence (348 months)
    data_clean2 = data_clean2.applymap(lambda x: replace_life(x)).copy()
    convert = ['birth_year', 'admission_year', 'year_prison_release']
    for col in convert:
        data_clean2[col] = pd.to_datetime(data_clean2[col], format='%Y')
    return data_clean2


def get_gender_race():
    """cleans and transforms raw data to clean"""
    # drop absent/single value columns and values
    data = pd.read_csv('./data/da26521-0003.tsv', sep='\t', header=0, low_memory=False)
    data = data.applymap(lambda x: replace_missing(x)).copy()
    absent_columns = list_of_absent_data_columns(data)
    data.drop(columns=absent_columns, inplace=True)
    singular_variables = make_singular_variable_list(data)
    data.drop(columns=singular_variables, inplace=True)
    data = data.dropna()
    data_clean = clean_target(data)
    # create dummy categoricals
    data_clean.rename(index=str, columns=variable_names, inplace=True)
    gender_race = data_clean[['sex', 'race']]
    return gender_race
