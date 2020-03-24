import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import statsmodels.api as sm
from scipy.integrate import odeint
from patsy import dmatrices

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
    ax.plot(days_fut_fmt, predict_fut, 'b', alpha=0.5, lw=2, label='Predicted')
    ax.plot(days_fut_fmt, response_fut, 'r', alpha=0.5, lw=2, label='Observed')


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

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# Data wrangling
@st.cache
def load_data():
    data = pd.read_excel('data/Covid-19_data.xlsx')
    data['day'] = [x.day for x in data['dia']]
    rel = pd.read_excel('data/Relacion_CCAA_CPROV.xlsx',sheet_name='Rel_CCAA_Name')
    ccaa_dict = {c:i for c,i in zip(rel['CCAA_Name'],rel['CCAA'])}
    n_prov = len(ccaa_dict)

    pob = pd.read_excel('data/Covid-19_data.xlsx',sheet_name='INE_Poblacion')
    data_pob = data.merge(pob,on=['CCAA','CCAA_Name'],how='left')
    data_pob['pct_casos_pob'] = data_pob['total_casos']/data_pob['pob']
    data_pob['pct_death_casos'] = data_pob['deaths']/data_pob['total_casos']
    data_pob['pct_death_pob'] = data_pob['deaths']/data_pob['pob']
    return data, rel, ccaa_dict, n_prov, pob, data_pob

st.sidebar.title('Cuadro de mando')
section_ind = st.sidebar.selectbox('Sección',['Acerca del proyecto','Documentación','Series temporales: Predicción a corto plazo','GLM: Estudio a corto plazo','SIR: Estudio a largo plazo'])

if section_ind=='Acerca del proyecto':
    st.title('Acerca del proyecto')
    '''Este es un proyecto para monitorizar y predecir la evolución del COVID-19 en España hecho por Juan José Vidal y Francisco Gabriel Morillas.
    Los datos han sido recopilados a partir de los informes de actualización del Ministerio de Sanidad.
    El software utilizado mayoritariamente Python. Los datos los recogemos en Excel, los manipulamos y predecimos con Python,
    presentamos el dashboard con Streamlit y publicamos la app en Heroku.
    '''

    st.title('Información sobre los autores')
    st.markdown('''**Juan José Vidal Llana**  
    Senior Pricing Analyst en Allianz España y Freelancer Strategy Consultant.  
    LinkedIn: [Juan José Vidal Llana](https://www.linkedin.com/in/juan-jose-vidal-llana/)  
    Mail: [xenxovidal@gmail.com](mailto:xenxovidal@gmail.com)  
    **Francisco Gabriel Morillas Jurado**  
    PDI Titular en la Universitat de València, Facultat d'Economia.  
    Mail: [francisco.morillas@uv.es](mailto:francisco.morillas@uv.es)  
    ''')
if section_ind=='Documentación':
    st.title('Documentación')
    st.write('Documentación del proyecto')

if section_ind=='Series temporales: Predicción a corto plazo':
    st.title('Series temporales: Predicción a corto plazo')
    st.write('Estudio de Fran')

if section_ind=='GLM: Estudio a corto plazo':
    st.title('Modelos GLM: Predicción a corto plazo')
    st.write('''En este apartado se estudia la evolución a corto plazo a nivel de CCAA según un modelo GLM Poisson con el día y la
    comunidad autónoma como únicos factores de riesgo. Hay que tener en cuenta que este modelo predecirá bien los próximos días pero
    no lo hará para un largo plazo.
    ''')
    data, rel, ccaa_dict, n_prov, pob, data_pob = load_data()
    # Model
    responses_display = {'deaths':'Fallecimientos', 'total_casos':'Casos observados'}
    response = st.selectbox('Respuesta',['deaths', 'total_casos'],format_func=lambda x: responses_display[x])
    data_model = data_pob[['day','CCAA',response]].copy()
    y, X = dmatrices(f'{response} ~ C(CCAA) + day', data=data_model, return_type='dataframe')

    mod = sm.GLM(y, X, family=sm.families.Poisson(), link=sm.families.links.logit)
    res = mod.fit()


    ca_name_short = st.selectbox('CCAA',list(ccaa_dict.keys()),key='short')
    plot_ccaa_curve(ca_name_short)



if section_ind=='SIR: Estudio a largo plazo':
    st.title('Modelos SIR: Predicción a largo plazo')
    st.write('''En este apartado se estudia la evolución a largo plazo a nivel de país o CCAA, según interese, y se ajusta el parámetro
    de contagio correspondiente al modelo SIR explicado en el apartado Documentación cada vez que se selecciona un periodo. Este modelo
    intenta predecir el pico de casos a largo terminio teniendo en cuenta una recuperación e inmunidad posterior por la sociedad.
    ''')
    data, rel, ccaa_dict, n_prov, pob, data_pob = load_data()
    ca_name_long = st.selectbox('CCAA',list(ccaa_dict.keys()),key='long')
    ca_number = ccaa_dict[ca_name_long]
    ca_name = rel.loc[rel['CCAA']==ca_number].reset_index()['CCAA_Name'][0]
    data_model = data_pob.loc[data_pob['CCAA']==ca_number].reset_index()[['CCAA', 'total_casos', 'pct_death_pob', 'pob', 'dia']]
    max_days = 201
    min_day = data_pob['dia'].min()
    days = [min_day]
    for i in range(max_days-1): 
        days.append(days[i] + datetime.timedelta(days=1))
    days_str = [d.strftime('%Y-%m-%d') for d in days]

    observed = list(data_model['pct_death_pob']) + list(np.repeat(None,max_days-len(data_model['pct_death_pob'])))

    n_obs = len(list(data_model['pct_death_pob']))
    beta_grid = np.linspace(0.1, 0.5, 41)

    rmse = []
    for b in beta_grid:
    # Total population, N.
        N = pob.loc[pob['CCAA']==ca_number].reset_index()['pob'][0]
        # Initial number of infected and recovered individuals, I0 and R0.
        I0 = data_model.loc[data_model['dia']==data_model['dia'].min()]['total_casos'][0]
        R0 = 0
        # Everyone else, S0, is susceptible to infection initially.
        S0 = N - I0 - R0
        # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
        beta = b
        gamma = 1/14
        # A grid of time points (in days)
        t = np.linspace(0, max_days-1, max_days)


        # Initial conditions vector
        y0 = S0, I0, R0
        # Integrate the SIR equations over the time grid, t.
        ret = odeint(deriv, y0, t, args=(N, beta, gamma))
        S, I, R = ret.T
        rmse.append(np.sqrt(sum((np.log(observed[n_obs-4:n_obs])-np.log(I[n_obs-4:n_obs]/N))**2)))

    val, idx = min((val, idx) for (idx, val) in enumerate(rmse))
    b_opt = beta_grid[idx]

    N = pob.loc[pob['CCAA']==ca_number].reset_index()['pob'][0]
    # Initial number of infected and recovered individuals, I0 and R0.
    I0 = data_model.loc[data_model['dia']==data_model['dia'].min()]['total_casos'][0]
    R0 = 0
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    beta = b_opt
    gamma = 1/14
    # A grid of time points (in days)
    t = np.linspace(0, max_days-1, max_days)


    # Initial conditions vector
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T

    st.write('El parámetro ajustado de contagio es: {:.2f}'.format(b_opt))
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w',figsize=(12,9))
    ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(days_str, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(days_str, I/N, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(days_str, R/N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.plot(days_str,observed, 'black', alpha=1, lw=3, label='Observed Infected')
    ax.set_xlabel('Time/days')
    ax.set_ylabel('% People')
    ax.set_ylim(0,1.1)
    #ax.set_ylim(0,0.001)
    #ax.set_xlim(0,20)
    loc = plticker.MultipleLocator(base=10) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=2,rotation=45)
    plt.xticks(ha='right')
    ax.grid(b=True, which='major', c='grey', lw=0.2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.title('{}: Prediction of the evolution of the COVID-19'.format(ca_name_long))
    st.pyplot()