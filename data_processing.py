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

def give_birth():
    """
    Return the birth rate per day per total population
    """
    return 53343/365/total_pop

def make_vaccine_csv():

    data = pd.DataFrame({
        "week": range(1, 53),
        "doses_administered": ""
    })
    data.to_csv("data/alberta_measles_doses_2025.csv", index=False)

def give_params_2():
    """
    Return empirical parameters as a dictionary for 2 group model
    """

    beta = np.array([[1.7280,6.686],[0.3688,9.9163]])
    beta = beta.flatten()
    gamma = np.array([1/8,1/8])
    delta = (1 / 1336) * (1 / 8)
    delta = np.array([delta,0])
    nu_ = give_nu()
    pop_1,pop_2 = data_processing()
    vpop_1, vpop_2 = 0.72*pop_1,0.91*pop_2
    infpop_1, infpop_2 = give_I()
    suspop_1,suspop_2 = pop_1 - infpop_1 - vpop_1, pop_2 - infpop_2 - vpop_2
    nu = np.array([0.9*nu_/suspop_1,0.1*nu_/suspop_2])
    params = {'beta_1': beta[0],'beta_2' : beta[1],'beta_3':beta[2],'beta_4':beta[3],'gamma_1' : gamma[0],'gamma_2' : gamma[1], 'delta' : delta[0], 'nu_1' : nu[0],'nu_2':nu[1]}
    return params


