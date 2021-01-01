__author__ = 'lucabasa'
__version__ = '0.0.1'

import numpy as np 
import pandas as pd
from pandas.api.types import CategoricalDtype


def total_question(data, question):
    qst_lst = [col for col in data if question+'_' in col]
    data[question+'_total'] = len(qst_lst) - data[qst_lst].isna().sum(axis=1)
    
    return data

def clean_data(data):
    data['Q4'] = data['Q4'].replace({"Some college/university study without earning a bachelor’s degree": 'College, no degree', 
                    'No formal education past high school': 'No formal education', 
                                 'I prefer not to answer': 'No Answer'}).fillna('No Answer')
    cat_type = CategoricalDtype(categories=["No Answer", "No formal education", "Professional degree", 
                                            "College, no degree", "Bachelor’s degree", 
                                            "Master’s degree", "Doctoral degree"], ordered=True)
    data['Q4'] = data['Q4'].astype(cat_type)

    income_dict = {'$0-999': 500, '10,000-14,999': 12500, '1,000-1,999': 1500, '100,000-124,999': 112500, 
                   '40,000-49,999': 45000, '30,000-39,999': 35000, '50,000-59,999': 55000, '5,000-7,499': 6250, 
                   '15,000-19,999': 17500, '60,000-69,999': 65000, '20,000-24,999': 22500, '70,000-79,999': 75000, 
                   '7,500-9,999': 8250, '150,000-199,999': 175000, '2,000-2,999': 2500, '125,000-149,999': 137500, 
                   '25,000-29,999': 27500, '90,000-99,999': 95000, '4,000-4,999': 4500, '80,000-89,999': 85000, 
                   '3,000-3,999': 3500, '200,000-249,999': 225000, '300,000-500,000': 400000, '> $500,000': 500000, 
                   '250,000-299,999': 275000}

    data['Q24_num'] = pd.to_numeric((data['Q24'].map(income_dict))) # .fillna(0)

    data['Q6'] = data['Q6'].replace({'I have never written code': '0 years'}).fillna('0 years')
    exp_cats = CategoricalDtype(categories=["0 years", "< 1 years", "1-2 years",
                                            "3-5 years", "5-10 years", 
                                            "10-20 years", "20+ years"], ordered=True)
    data['Q6'] = data['Q6'].astype(exp_cats)

    # for better merge with the continents
    data['Q3'] = data['Q3'].replace({'Iran, Islamic Republic of...': 'Iran (Islamic Republic of)', 
                                     'Russia': 'Russian Federation', 
                                     'Taiwan': 'Taiwan, Province of China', 
                                     'South Korea': 'Korea (Republic of)', 
                                     'Republic of Korea': 'Korea (Republic of)'})

    continents = pd.read_csv('data/countryContinent.csv', encoding = 'ISO-8859-1')
    data = pd.merge(data, continents[['country', 'continent', 'sub_region']], left_on='Q3', right_on='country', how='left')


    data['Q3'] = data['Q3'].replace({'United States of America': 'USA', 
                                     'United Kingdom of Great Britain and Northern Ireland': 'UK', 
                                     'Russian Federation': 'Russia', 
                                     'Korea (Republic of)': 'S. Korea', 
                                     'Taiwan, Province of China': 'Taiwan'})

    data['sub_region'] = (data.sub_region.str.replace('ern', '')
                          .str.replace('South', 'S')
                          .str.replace('North', 'N')
                          .str.replace('West', 'W')
                          .str.replace('East', 'E')
                          .str.replace('Central', 'C')
                          .str.replace('Australia and New Zealand', 'Aus and NZ'))
    sub_reg = CategoricalDtype(categories=["E Africa", "N Africa", "S Africa", "W Africa",
                                            "C America", "N America", "S America",
                                            "E Asia", "S-E Asia", "S Asia", "W Asia", "Aus and NZ",
                                           "E Europe", "N Europe", "S Europe", "W Europe"], ordered=True)
    data['sub_region'] = data['sub_region'].astype(sub_reg)

    exp_cats = CategoricalDtype(categories=["0 years", "Under 1 year", "1-2 years", "2-3 years",
                                            "3-4 years", "4-5 years", "5-10 years", 
                                            "10-20 years", "20 or more years"], ordered=True)
    data['Q15'] = data['Q15'].replace({'I do not use machine learning methods': '0 years'}).fillna('0 years')

    data['Q15'] = data['Q15'].astype(exp_cats)

    emp_cats = CategoricalDtype(categories=["0-49 employees", "50-249 employees", 
                                            "250-999 employees", "1000-9,999 employees",
                                            "10,000 or more employees"], ordered=True)
    data['Q20'] = data['Q20'].astype(emp_cats)

    size_cats = CategoricalDtype(categories=["0", "1-2", 
                                            "3-4", "5-9", "10-14", '15-19',
                                            "20+"], ordered=True)
    data['Q21'] = data['Q21'].astype(size_cats)

    data['Q22'] = data['Q22'].replace({'We are exploring ML methods (and may one day put a model into production)': 'Exploring', 
                                       'No (we do not use ML methods)': "No / I don't know", 'I do not know': "No / I don't know",
                                      "We have well established ML methods (i.e., models in production for more than 2 years)": "In prod. for more than 2 years", 
                                       "We recently started using ML methods (i.e., models in production for less than 2 years)": "In prod. for less than 2 years", 
                                       "We use ML methods for generating insights (but do not put working models into production)": 'Insights'})
    ml_cat = CategoricalDtype(categories=["No / I don't know", 'Exploring', 
                                            'Insights', "In prod. for less than 2 years", 
                                          "In prod. for more than 2 years"], ordered=True)
    data['Q22'] = data['Q22'].astype(ml_cat)
    return data