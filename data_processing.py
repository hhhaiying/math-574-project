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

def give_nu():
    """
    Return an averaged nu from weekly vaccine data in units per day (not divided by susceptibles yet)
    """
    data = pd.read_csv("data/alberta_measles_doses_2025.csv")
    avg = data['doses_administered'].mean()
    return (avg/7)/total_pop

def give_I():
    """
    Return proportion of population infected with measles from 2025 data
    """
    i_1 = 580
    i_2 = 1428
    i_1 /= total_pop
    i_2 /= total_pop
    return i_1,i_2

def make_vaccine_csv():

    data = pd.DataFrame({
        "week": range(1, 53),
        "doses_administered": ""
    })
    data.to_csv("data/alberta_measles_doses_2025.csv", index=False)


print(give_nu())
