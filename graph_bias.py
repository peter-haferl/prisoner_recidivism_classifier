#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd
from clean_explore import get_gender_race

def plot_bias(set_type=None, pos_values=None, neg_values=None, size=None):
    '''
    Product bar plot percentages of false postives/negative 
    
    set_type = 'gender' or 'race'
    pos_values = list of false positive values
    neg_values = list of false negative values
    size = tuple of figure size
    '''
    if set_type == 'gender':
        x_cats = ['Men', 'Women']
    elif set_type == 'race':
        x_cats = ['White', 'Black', 'American Indian', 'Asian', 'Other']
    else:
        raise('Not a valid type')
    
    fig, ax = plt.subplots(2,1, figsize=size)
    
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 18}
    
    plt.rc('font', **font)
    ax[0].bar(x=x_cats, height=pos_values)
    ax[0].set_title('False Positive', fontdict=font)
    ax[0].set_ylabel('Percentage False Postive', fontdict=font)
    
    ax[1].bar(x=x_cats, height=neg_values)
    ax[1].set_title('False Negative', fontdict=font)
    ax[1].set_ylabel('Percentage False Negative', fontdict=font)
    
    plt.subplots_adjust(hspace = 0.25)
    
    return fig


def analyze_bias(y_test, y_test_pred):
    ''' 
    Analyze gender and racial bias of classification model
        
    y_test = Pandas Series of real test target values -- !!! With indexing corresponding to original dataset !!!
    y_test_pred = Array readout of sklearn model .predit() method 
    '''
    # Get race
    'gender_race = get_gender_race_info()'
    
    # Merge predict data with test data and racial information
    preds = pd.concat([y_test, pd.Series(y_test_pred, name='pred', index=y_test.index)], axis=1, ignore_index=False)
    preds = pd.merge(preds, gender_race, left_index=True, right_index=True)
    
    # Gender Bias
    sex_false_pos = preds.loc[(preds['outcome'] == 0) & (preds['pred'] == 1)]['sex']
    men_false_pos = sum(sex_false_pos == 1)/sum(preds['sex'] == 1)
    women_false_pos = sum(sex_false_pos == 2)/sum(preds['sex'] == 2)
    
    sex_false_neg = preds.loc[(preds['outcome'] == 1) & (preds['pred'] == 0)]['sex']
    men_false_neg = sum(sex_false_neg == 1)/sum(preds['sex'] == 1)
    women_false_neg = sum(sex_false_neg == 2)/sum(preds['sex'] == 2)
    
    gender_pos_values = [men_false_pos,women_false_pos]
    gender_neg_values = [men_false_neg,women_false_neg]
    
    
    plot_bias(set_type='gender', pos_values=gender_pos_values, neg_values=gender_neg_values, size=(10,10))
    
    # Race Bias
    
    racial_false_pos = preds.loc[(preds['outcome'] == 0) & (preds['pred'] == 1)]['race']
    white_false_pos = sum(racial_false_pos == 1)/sum(preds['race'] == 1)
    black_false_pos = sum(racial_false_pos == 2)/sum(preds['race'] == 2)
    americanindian_false_pos = sum(racial_false_pos == 3)/sum(preds['race'] == 3)
    asian_false_pos = sum(racial_false_pos == 4)/sum(preds['race'] == 4)
    other_false_pos = sum(racial_false_pos == 6)/sum(preds['race'] == 6)
    
    racial_false_neg = preds.loc[(preds['outcome'] == 0) & (preds['pred'] == 1)]['race']
    white_false_neg = sum(racial_false_pos == 1)/sum(preds['race'] == 1)
    black_false_neg = sum(racial_false_pos == 2)/sum(preds['race'] == 2)
    americanindian_false_neg = sum(racial_false_pos == 3)/sum(preds['race'] == 3)
    asian_false_neg = sum(racial_false_pos == 4)/sum(preds['race'] == 4)
    other_false_neg = sum(racial_false_pos == 6)/sum(preds['race'] == 6)

    race_pos_values = [white_false_pos, black_false_pos, americanindian_false_pos, asian_false_pos, other_false_pos]
    race_neg_values = [white_false_neg, black_false_neg, americanindian_false_neg, asian_false_neg, other_false_neg]
    
    plot_bias(set_type='race', pos_values=race_pos_values, neg_values=race_neg_values, size=(12,10))

