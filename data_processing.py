import pandas as pd
import numpy as np

data = pd.read_csv('data/population.csv')
total_pop = data['Population'].sum(axis=0)

def data_processing():
    """
    Give proportion population from 0 to 5 and 5 - 99
    """
    age_05 = data['Population'][0:2].sum()
    rest = total_pop- age_05
    age_05 /= total_pop
    rest /= total_pop
    return age_05,rest

def give_nu(nu):
    """
    Return an averaged nu from population data
    """
    pop_1,pop_2 = data_processing()
    new = nu[0]*pop_1 + nu[1]*pop_2
    return new

def give_I():
    """
    Return proportion of population infected with measles from 2025 data
    """
    i_1 = 580
    i_2 = 1428
    i_1 /= total_pop
    i_2 /= total_pop
    return i_1,i_2

