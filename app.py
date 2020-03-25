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
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=2,rotation=45)
    plt.xticks(ha='right')
    ax.grid(b=True, which='major', c='grey', lw=0.2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.title('{}: Prediction of the evolution of the {}, COVID-19'.format(ca,response))
    st.pyplot()
    st.write(pd.DataFrame({'Day':days_fut_fmt,'Observed':response_fut,'Predictions':[int(p) for p in predict_fut]}))

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def predict_geoserie(d,remove,post_days,geom_days,day_ini):
    if remove>0:
        d = d[:-remove]
    d_long = d + list(np.repeat(None,post_days))
    
    n_obs = len(d)
    n_obs_long = n_obs + post_days
    
    days_long = [day_ini]
    for c in range(n_obs-1 + post_days):
        days_long.append(days_long[c] + datetime.timedelta(days=1))
    days = days_long[:-post_days]
    dia_count_long = list(range(1,n_obs+1+post_days))

    tend_long = [np.log(d-5) for d in dia_count_long[geom_days:]]
    tend = tend_long[:-post_days]

    d_diff = [d[i]/d[i-1] for i in range(1,n_obs)]

    d_geomean = [(d_diff[i]*d_diff[i+1]*d_diff[i+2]*d_diff[i+3]*d_diff[i+4])**(1/geom_days) for i in range(1,n_obs-geom_days)]
    i_last = n_obs-geom_days
    d_geomean.append((d_diff[i_last]*d_diff[i_last+1]*d_diff[i_last+2]*d_diff[i_last+3])**(1/(geom_days-1)))

    b = sum([(t-np.mean(tend))*(g-np.mean(d_geomean)) for t,g in zip(tend,d_geomean)])/sum([(t-np.mean(tend))**2 for t,g in zip(tend,d_geomean)])
    a = np.mean(d_geomean)-b*np.mean(tend)

    mult_fut = [a + b*t for t in tend_long[(-post_days-1):-1]]

    preds_fut = [d[n_obs-1]*d_geomean[-1]] + list(np.repeat(None, post_days-1))

    for i in range(1,post_days):
        preds_fut[i] = preds_fut[i-1]*mult_fut[i]

    preds_fut_long = list(np.repeat(None,len(d))) + preds_fut
    return d_long, preds_fut_long, days_long

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

st.sidebar.title('Índice')
section_ind = st.sidebar.radio('',['Introducción','Series temporales: Estudio a corto plazo','GLM: Estudio a corto plazo','SIR: Estudio a largo plazo','Documentación','Acerca del proyecto'])

if section_ind=='Introducción':
    st.title('Proyección sobre la evolución de la incidencia del virus COVID-19')
    st.markdown('''
    ## Motivación
    Ante la situación de alarma actual en relación al denominado COVID-19, se han
    puesto en marcha diferentes iniciativas que tratan de anticipar algunos de los efectos
    negativos que la pandemia actual genera, de esta manera es posible planificar
    diferentes escenarios y tomar decisiones con el objetivo de paliar las consecuencias
    más negativas de esta enfermedad. Así, el CEMat se encarga de coordinar las
    iniciativas de la comunidad matemática española relacionadas con la crisis creada
    por el COVID-19. Así, un grupo amplio de investigadores trata de encontrar una
    respuesta al problema actual de salud pública mencionado.  
    En particular, desde este grupo de investigación de la Facultat d’Economia de la
    Universitat de València, también tratamos de realizar una pequeña contribución,
    teniendo en cuenta las dificultades que el problema analizado presenta y que, por
    tanto, limita la respuesta de los diferentes modelos matemáticos que se utilizan.
    ''')

    st.markdown('''
    ## Introducción Metodológica
    Existen diferentes maneras de afrontar este tipo de problemas: (i) con modelos
    predictivos, haciendo uso de técnicas estadísticas como los denominados GLM
    (Modelo Lineal Generalizado); (ii) con modelos epidemiológicos establecidos, como
    los denominados SIR (Susceptibles-Infectados-Recuperados) y otros modelos
    derivados de este; y, (iii) técnicas de Series Temporales, en las que se analiza una o
    más variables de interés y se establece una relación estructural de evolución temporal
    que se asume ‘persistente’ en el tiempo. (iv) Por supuesto, existen otros enfoques
    pero he pasado a describir los más utilizados actualmente.  
    El grupo de investigación está trabajando en los tres tipos mencionados (i) a (iii).
    ''')  
    st.markdown('''
    ## Frecuencia de actualización del análisis
    Los resultados se amplían diariamente con los valores observados y se recalibran los
    parámetros del modelo y los ajustes utilizados.
    ''')
    st.markdown('''
    ## Horizonte de predicción y variables analizadas
    Las estimaciones son útiles en el corto plazo (1-3 días), las variables analizadas son
    el número de fallecidos, y el número de ingresos en UCI, para el total acumulado en
    el conjunto del territorio nacional.
    ''')    
    st.markdown('''
    ## Datos y Fuentes de Información
    Los datos utilizados son los publicados por el Gobierno Español, aunque en la fase
    inicial se utilizaron los datos recopilados y depurados por el grupo de trabajo
    Datadista (Github) y los proporcionados por el Johns Hopkins CSSE.
    ''')

if section_ind=='Series temporales: Estudio a corto plazo':
    st.title('Series temporales: Estudio a corto plazo')
    data, rel, ccaa_dict, n_prov, pob, data_pob = load_data()
    data_pob_agg = data_pob[['dia', 'total_casos','deaths', 'pob']].groupby('dia').agg('sum').reset_index()
    d = list(data_pob_agg['deaths'])
    post_days = 3
    geom_days = 5
    day_ini = data_pob_agg['dia'].min()

    d_long_0, preds_fut_long_0, days_long = predict_geoserie(d, 0, post_days, geom_days, day_ini)
    d_long_1, preds_fut_long_1, _ = predict_geoserie(d, 1, post_days+1, geom_days, day_ini)
    d_long_2, preds_fut_long_2, _ = predict_geoserie(d, 2, post_days+2, geom_days, day_ini)
    d_long_3, preds_fut_long_3, _ = predict_geoserie(d, 3, post_days+3, geom_days, day_ini)

    fig = plt.figure(facecolor='w',figsize=(12,9))
    ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(days_long, d_long_0, 'black', alpha=0.5, lw=3, label='Observado hasta el {}'.format(days_long[(-post_days-1)].strftime('%Y-%m-%d')))
    ax.plot(days_long, preds_fut_long_0, 'o-r', alpha=0.5, lw=2, label='Predicción con datos hasta el {}'.format(days_long[(-post_days-1)].strftime('%Y-%m-%d')))
    ax.plot(days_long, preds_fut_long_1, 'o-b', alpha=0.5, lw=2, label='Predicción con datos hasta el {}'.format(days_long[(-1-post_days-1)].strftime('%Y-%m-%d')))
    ax.plot(days_long, preds_fut_long_2, 'o-y', alpha=0.5, lw=2, label='Predicción con datos hasta el {}'.format(days_long[(-2-post_days-1)].strftime('%Y-%m-%d')))
    ax.plot(days_long, preds_fut_long_3, 'o-g', alpha=0.5, lw=2, label='Predicción con datos hasta el {}'.format(days_long[(-3-post_days-1)].strftime('%Y-%m-%d')))
    ax.set_xlabel('Día')
    ax.set_ylabel('Fallecimientos')
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=2,rotation=45)
    plt.xticks(ha='right')
    ax.grid(b=True, which='major', c='grey', lw=0.5, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.title('Datos observados y predicción')
    plt.suptitle('Número de fallecidos acumulado por día',size=20)
    st.pyplot()

    st.table(pd.DataFrame({'Día':[d_f.strftime('%Y-%m-%d') for d_f in days_long],
                        'Observado': d_long_0,
                        'Pred_3':[int(x) if x is not None else None for x in preds_fut_long_3],
                        'Pred_2':[int(x) if x is not None else None for x in preds_fut_long_2],
                        'Pred_1':[int(x) if x is not None else None for x in preds_fut_long_1],
                        'Pred_0':[int(x) if x is not None else None for x in preds_fut_long_0]}))

if section_ind=='GLM: Estudio a corto plazo':
    st.title('Modelos GLM: Estudio a corto plazo')
    st.write('''En este apartado se estudia la evolución a corto plazo a nivel de CCAA según un modelo GLM Poisson con el día y la
    comunidad autónoma como únicos factores de riesgo. Hay que tener en cuenta que este modelo predecirá bien los próximos días pero
    no lo hará para un largo plazo, ya que en ningún momento se tiene en cuenta una futura bajada del contagio.
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
    st.title('Modelos SIR: Estudio a largo plazo')
    st.write('''En este apartado se estudia la evolución a largo plazo a nivel de país o CCAA, según interese, y se ajusta el parámetro
    de contagio correspondiente al modelo SIR explicado en el apartado Documentación cada vez que se selecciona un periodo. Este modelo
    intenta predecir el pico de casos a largo terminio teniendo en cuenta una recuperación e inmunidad posterior por la sociedad.
    ''')
    
    all_ccaa = st.checkbox('Todo el país / CCAA', True)

    max_days = 201

    if all_ccaa:
        data, rel, ccaa_dict, n_prov, pob, data_pob = load_data()
        data_pob_agg = data_pob[['dia', 'total_casos','deaths', 'pob']].groupby('dia').agg('sum').reset_index()
        data_pob_agg['pct_death_pob'] = data_pob_agg['deaths']/data_pob_agg['pob']
        observed = list(data_pob_agg['pct_death_pob']) + list(np.repeat(None,max_days-len(data_pob_agg['pct_death_pob'])))
        n_obs = len(list(data_pob_agg['pct_death_pob']))
        # Total population, N.
        N = data_pob_agg['pob'][0]
        # Initial number of infected and recovered individuals, I0 and R0.
        I0 = data_pob_agg.loc[data_pob_agg['dia']==data_pob_agg['dia'].min()]['total_casos'][0]
        ca_name_long = 'España'

    else:
        data, rel, ccaa_dict, n_prov, pob, data_pob = load_data()
        ca_name_long = st.selectbox('CCAA',list(ccaa_dict.keys()),key='long')
        ca_number = ccaa_dict[ca_name_long]
        ca_name = rel.loc[rel['CCAA']==ca_number].reset_index()['CCAA_Name'][0]
        data_model = data_pob.loc[data_pob['CCAA']==ca_number].reset_index()[['CCAA', 'total_casos', 'pct_death_pob', 'pob', 'dia']]
        observed = list(data_model['pct_death_pob']) + list(np.repeat(None,max_days-len(data_model['pct_death_pob'])))
        n_obs = len(list(data_model['pct_death_pob']))
        # Total population, N.
        N = pob.loc[pob['CCAA']==ca_number].reset_index()['pob'][0]
        # Initial number of infected and recovered individuals, I0 and R0.
        I0 = data_model.loc[data_model['dia']==data_model['dia'].min()]['total_casos'][0]

    min_day = data_pob['dia'].min()
    days = [min_day]
    for i in range(max_days-1): 
        days.append(days[i] + datetime.timedelta(days=1))
    days_str = [d.strftime('%Y-%m-%d') for d in days]
    beta_grid = np.linspace(0.1, 0.5, 41)

    R0 = 0
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0
    # Mean recovery rate, gamma, (in 1/days).
    gamma = 1/14
    # A grid of time points (in days)
    t = np.linspace(0, max_days-1, max_days)
    # Initial conditions vector
    y0 = S0, I0, R0

    rmse = []
    for b in beta_grid:
        # Contact rate, beta
        beta = b
        # Integrate the SIR equations over the time grid, t.
        ret = odeint(deriv, y0, t, args=(N, beta, gamma))
        S, I, R = ret.T
        rmse.append(np.sqrt(sum((np.log(observed[n_obs-4:n_obs])-np.log(I[n_obs-4:n_obs]/N))**2)))

    val, idx = min((val, idx) for (idx, val) in enumerate(rmse))
    b_opt = beta_grid[idx]
    beta = b_opt

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

    val_inf, idx_inf = max((val, idx) for (idx, val) in enumerate(I))
    st.write('Se espera que en {}, el {} se alcance el pico máximo de infectados con un {:.2%} de la población afectada.'.format(ca_name_long,days_str[idx_inf],val_inf/N))

if section_ind=='Documentación':
    st.title('Documentación')
    st.markdown('''
    En relación a la metodología utilizada en las **series temporales** se ha seguido el siguiente esquema:  

    1. Se consideran Tasas de Variación entre valores diarios consecutivos.
    2. Para estimar la tendencia se inicia el proceso mediante una media geométrica
    móvil.
    3. La estimación de la tendencia se realiza mediante un ajuste funcional de la
    serie obtenida en el paso anterior. En este caso logarítmico.
    4. Se utiliza la estimación anterior para predecir los valores en momentos
    futuros.
    5. Con los valores de tendencia y los valores observados, mediante
    encadenamiento, se obtienen los valores que conforman la predicción.
    6. La estimación de los errores se basa en el error del ajuste funcional y no
    aparece en este documento.
    7. La generación de escenarios alternativos y plausibles se está diseñando y
    ajustando para que pueda ser utilizado en la práctica.
    ''')

    st.markdown('''
    En relación con los **Modelos Lineares Generalizados**, estos son una generalización flexible de la regresión lineal ordinaria 
    que permite variables de respuesta que tienen modelos de distribución de errores distintos de una distribución normal. 
    El GLM generaliza la regresión lineal al permitir que el modelo lineal esté relacionado con la variable de respuesta a 
    través de una función de enlace y al permitir que la magnitud de la varianza de cada medición sea una función de su valor predicho. 
    [+info](https://es.wikipedia.org/wiki/Modelo_lineal_generalizado)
    ''')

    st.markdown('''
    Respecto a los **modelos SIR**, son uno de los modelos epidemiológicos más simples capaces de capturar muchas de las características 
    típicas de los brotes epidémicos. El nombre del modelo proviene de las iniciales S (población susceptible), I (población infectada) 
    y R (población recuperada). El modelo relaciona las variaciones las tres poblaciones (Susceptible, Infectada y Recuperada) 
    a través de la tasa de infección y el período infeccioso promedio. [+info](https://es.wikipedia.org/wiki/Modelo_SIR)
    ''')

if section_ind=='Acerca del proyecto':
    st.title('Acerca del proyecto')
    '''Este es un proyecto para monitorizar y predecir la evolución del COVID-19 en España desarrollado por Juan José Vidal y Francisco Gabriel Morillas.
    Los datos han sido recopilados a partir de los informes de actualización del Ministerio de Sanidad.
    El software utilizado mayoritariamente Python. Los datos los recogemos en Excel, los manipulamos y predecimos con Python,
    presentamos el dashboard con Streamlit y publicamos la app en Heroku.
    '''

    st.title('Información sobre los autores')
    st.markdown('''**Juan José Vidal Llana**  
    Senior Pricing Analyst en Allianz España y Freelancer Strategy Consultant.  
    LinkedIn: [Juan José Vidal Llana](https://www.linkedin.com/in/juan-jose-vidal-llana/)  
    Mail: [xenxovidal@gmail.com](mailto:xenxovidal@gmail.com)  
    ''')
    st.markdown('''**Francisco Gabriel Morillas Jurado**  
    PDI Titular en la Universitat de València, Facultat d'Economia.  
    LinkedIn: [Francisco Gabriel Morillas Jurado](https://www.linkedin.com/in/fracisco-gabriel-morillas-jurado-4a5290a6/?originalSubdomain=es)  
    Mail: [francisco.morillas@uv.es](mailto:francisco.morillas@uv.es)  
    ''')