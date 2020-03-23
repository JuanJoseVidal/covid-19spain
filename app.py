import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from patsy import dmatrices
# http://10.188.166.68:8501

def predict_model(ca,day):
    pred = [1] + list(np.repeat(0,18)) + [day]
    pred[ccaa_dict[ca]-1] = 1
    return float(res.predict(pred))


def plot_ccaa_curve(ca):
    data_plot = data.query(f'CCAA_Name=="{ca}"')[['day',response]]
    response_fut = list(data_plot[response]) + list(np.repeat(None,5))
    days_fut = list(data_plot['day']) + list(range(data_plot['day'].max()+1,data_plot['day'].max()+6))
    predict_fut = [predict_model(ca,d) for d in days_fut]


    max_days = len(days_fut)
    min_day = datetime.date(2020,3,min(days_fut))
    days = [min_day]
    for i in range(max_days-1): 
        days.append(days[i] + datetime.timedelta(days=1))
    days_fut_fmt = [d.strftime('%Y-%m-%d') for d in days]

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w',figsize=(12,9))
    ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(days_fut_fmt, predict_fut, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(days_fut_fmt, response_fut, 'r', alpha=0.5, lw=2, label='Infected')


    ax.set_xlabel('Día')
    ax.set_ylabel(response)
    #ax.set_ylim(0,1.1)
    #ax.set_ylim(0,0.001)
    #ax.set_xlim(0,20)
    #loc = plticker.MultipleLocator(base=10) # this locator puts ticks at regular intervals
    #ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=2,rotation=45)
    plt.xticks(ha='right')
    ax.grid(b=True, which='major', c='grey', lw=0.2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.title('{}: Prediction of the evolution of the {}, COVID-19'.format(ca,response))
    #plt.savefig(p+'export/Evolution_Spain.png')
    st.pyplot()

# Data wrangling
data = pd.read_excel('Covid-19_data.xlsx')
data['day'] = [x.day for x in data['dia']]
rel = pd.read_excel('maps/Relacion_CCAA_CPROV.xlsx',sheet_name='Rel_CCAA_Name')
ccaa_dict = {c:i for c,i in zip(rel['CCAA_Name'],rel['CCAA'])}
n_prov = len(ccaa_dict)

pob = pd.read_excel('Covid-19_data.xlsx',sheet_name='INE_Poblacion')
data_pob = data.merge(pob,on=['CCAA','CCAA_Name'],how='left')
data_pob['pct_casos_pob'] = data_pob['total_casos']/data_pob['pob']
data_pob['pct_death_casos'] = data_pob['deaths']/data_pob['total_casos']
data_pob['pct_death_pob'] = data_pob['deaths']/data_pob['pob']

st.title('Predicción de número de casos o fallecimientos a corto plazo')

# Model
response = st.selectbox('Respuesta',['deaths', 'total_casos'])
data_model = data_pob[['day','CCAA',response]].copy()
y, X = dmatrices(f'{response} ~ C(CCAA) + day', data=data_model, return_type='dataframe')

mod = sm.GLM(y, X, family=sm.families.Poisson(), link=sm.families.links.logit)
res = mod.fit()


ca = st.selectbox('CCAA',list(ccaa_dict.keys()))
plot_ccaa_curve(ca)




# st.title('Predicción de evolución a largo plazo')