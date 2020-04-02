import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib import cm
import statsmodels.api as sm
from scipy.integrate import odeint
from patsy import dmatrices

# Data wrangling
@st.cache
def load_data():
    data = pd.read_excel('data/Covid-19_data.xlsx')
    data = data.loc[data['to_study']=='Yes']
    data['day'] = [transform_day_glm(x) for x in data['dia']]
    rel = pd.read_excel('data/Relacion_CCAA_CPROV.xlsx',sheet_name='Rel_CCAA_Name')
    rel = rel.loc[rel['CCAA'] < 18]
    ccaa_dict = {c:i for c,i in zip(rel['CCAA_Name'],rel['CCAA'])}
    n_prov = len(ccaa_dict)

    pob = pd.read_excel('data/Covid-19_data.xlsx',sheet_name='INE_Poblacion')
    data_pob = data.merge(pob,on=['CCAA','CCAA_Name'],how='left')
    data_pob['pct_casos_pob'] = data_pob['total_casos']/data_pob['pob']
    data_pob['pct_death_casos'] = data_pob['deaths']/data_pob['total_casos']
    data_pob['pct_death_pob'] = data_pob['deaths']/data_pob['pob']
    return data, rel, ccaa_dict, n_prov, pob, data_pob

@st.cache
def load_data_sexage():
    data_sexage = pd.read_excel('data/Covid-19_data_sexage.xlsx')
    data_sexage = data_sexage.query('gender!="Tot"')
    return data_sexage

def transform_day_glm(d):
    if d.month <= 3:
        return d.day
    else:
        return d.day + 31

def extract_diffs(var, dia_study):
    dia_study_ant = dia_study - datetime.timedelta(days=1)
    return int(data_pob_agg.query(f'dia=="{dia_study}"').reset_index()[var][0] - \
            data_pob_agg.query(f'dia=="{dia_study_ant}"').reset_index()[var][0])

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(int(height)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def extract_diffs_ccaa(var, dia_study, ccaa):
    dia_study_ant = dia_study - datetime.timedelta(days=1)
    return data_pob.query(f'dia=="{dia_study}"&CCAA_Name=="{ccaa}"').reset_index()[var][0] - \
            data_pob.query(f'dia=="{dia_study_ant}"&CCAA_Name=="{ccaa}"').reset_index()[var][0]

def plot_letality_gender(gend):
    data_gnd = data_sexage.query(f'gender=="{gend}"')
    edades = data_gnd['edad'].unique()
    days_plot = data_gnd['dia'].unique()
    cmap = cm.get_cmap('Spectral',len(edades))
    newcolors = cmap(np.linspace(0, 1, len(days_plot)))

    fig = plt.figure(facecolor='w',figsize=(12,9))
    ax = fig.add_subplot(111, axisbelow=True)
    for d in range(len(days_plot)):
        death_loop = data_gnd.query(f'dia=="{days_plot[d]}"')['letalidad']
        ax.plot(edades, death_loop/100, 'black', alpha=0.5, lw=3, color=newcolors[d], label = f'Dia: "{pd.to_datetime(str(days_plot[d])).strftime("%Y-%m-%d")}"')
    ax.set_xlabel('Edad')
    ax.set_ylabel('Letalidad (%)')
    ax.grid(b=True, which='major', c='grey', lw=0.5, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    plt.title('Letalidad acumulada por día para {} según edad.'.format("hombres" if gend =='Masc' else "mujeres"))
    st.pyplot()

def color_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    return 'color: red' 

def remove_na(l,pct=False):
    if pct == True:
        return ['{:.2%}'.format(x) if x is not None else "" for x in l]
    else:
        return [str(int(x)) if str(x) != 'nan' and x != None  else "" for x in l]

def extract_only_fut_preds(i):
    return list(np.repeat(None,len(predict_fut[i])-3-i)) + [int(p) for p in predict_fut[i][(-3-i):]]

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def predict_model(res,ca,day):
    pred = [1] + list(np.repeat(0,16)) + [1] + [day**2] + [day]
    pred[ccaa_dict[ca]-1] = 1
    pred = pred + [p*day for p in pred[1:17]]
    return res.predict(pred)
    
@st.cache
def train_glm(response):
    predict_df = {}
    response_df = {}

    for ca in ccaa_dict.keys():
        res = []
        predict_fut_ccaa = []

        for i in range(5):
            n_days_test = i
            if i==0:
                data_red = data.copy()
            else:
                data_test = data.loc[data['dia']>=np.sort(data['dia'].unique())[-n_days_test]]
                data_red = data.loc[data['dia']<np.sort(data['dia'].unique())[-n_days_test]]

            # Model
            data_model = data_red[['day','CCAA',response, 'confin']].copy()
            y, X = dmatrices(f'{response} ~ I(day**2) + C(CCAA)*day + confin', data=data_model, return_type='dataframe')

            mod = sm.GLM(y, X, family=sm.families.Poisson(), link=sm.families.links.logit)
            res.append(mod.fit())
            data_plot = data_red.query(f'CCAA_Name=="{ca}"')[['day',response]]
            if i==0:
                response_df.update({ca:list(data_plot[response]) + list(np.repeat(None,3+i))})
            days_fut = list(data_plot['day']) + list(range(data_plot['day'].max()+1,data_plot['day'].max()+4+i))

            predict_fut_ccaa.append([predict_model(res[i],ca,d) for d in days_fut])
        predict_df.update({ca:predict_fut_ccaa})

    max_days = len(days_fut)
    min_day = datetime.date(2020,3,min(days_fut))
    days = [min_day]
    for i in range(max_days-1): 
        days.append(days[i] + datetime.timedelta(days=1))
    days_fut_fmt = [d.strftime('%Y-%m-%d') for d in days]


    predict_df_0 = pd.DataFrame({ca:predict_df[ca][0] for ca in list(ccaa_dict.keys())})
    predict_df_1 = pd.DataFrame({ca:predict_df[ca][1] for ca in list(ccaa_dict.keys())})
    predict_df_2 = pd.DataFrame({ca:predict_df[ca][2] for ca in list(ccaa_dict.keys())})
    predict_df_3 = pd.DataFrame({ca:predict_df[ca][3] for ca in list(ccaa_dict.keys())})
    predict_df_4 = pd.DataFrame({ca:predict_df[ca][4] for ca in list(ccaa_dict.keys())})
    predict_df_full = [predict_df_0, predict_df_1, predict_df_2, predict_df_3, predict_df_4]
    return predict_df_full, response_df, days_fut_fmt

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
    
st.sidebar.title('Índice')
section_ind = st.sidebar.radio('',['Introducción', 'Informe diario', 'Evolución por género', 'Series temporales: Estudio a corto plazo','GLM: Estudio a corto plazo','SIR: Estudio a largo plazo','Documentación','Acerca del proyecto'])

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

if section_ind=='Informe diario':
    st.title('Informe de evolución diaria.')
    data, rel, ccaa_dict, n_prov, pob, data_pob = load_data()
    data_pob_agg = data_pob[['dia', 'total_casos', 'deaths', 'hospit', 'ingr_UCI', 'pob']].groupby('dia').agg('sum').reset_index()
    data_pob_agg['pct_death_pob'] = data_pob_agg['deaths']/data_pob_agg['pob']
    ult_day = data_pob_agg['dia'].max()
    st.markdown('## Acumulado a día {}.'.format(ult_day.strftime('%Y-%m-%d')))
    st.write('Casos confirmados: ',int(data_pob_agg.query(f'dia=="{ult_day}"').reset_index()['total_casos'][0]))
    st.write('Fallecimientos: ',int(data_pob_agg.query(f'dia=="{ult_day}"').reset_index()['deaths'][0]))
    st.write('Casos hospitalizados: ',int(data_pob_agg.query(f'dia=="{ult_day}"').reset_index()['hospit'][0]))
    st.write('Ingresados en UCI: ',int(data_pob_agg.query(f'dia=="{ult_day}"').reset_index()['ingr_UCI'][0]))

    st.markdown('## Evolución diaria.')
    dia_select = [d for d in data_pob_agg['dia'][1:]]
    day_format_func = lambda x: x.strftime('%Y-%m-%d') 
    dia_study = st.selectbox('Día',dia_select,key='day_informe',index=len(dia_select)-1,format_func=day_format_func)

    st.write('Nuevos casos confirmados: ',extract_diffs('total_casos', dia_study))
    st.write('Nuevos fallecimientos: ',extract_diffs('deaths', dia_study))
    st.write('Nuevos casos hospitalizados: ',extract_diffs('hospit', dia_study))
    st.write('Nuevos ingresados en UCI: ',extract_diffs('ingr_UCI', dia_study))

    ca_name_inf = st.selectbox('CCAA',list(ccaa_dict.keys()),index=12,key='ca_informe')
    inf_data_ccaa = [extract_diffs_ccaa('total_casos', dia_study, ca_name_inf),\
    extract_diffs_ccaa('deaths', dia_study, ca_name_inf),\
    extract_diffs_ccaa('hospit', dia_study, ca_name_inf),\
    extract_diffs_ccaa('ingr_UCI', dia_study, ca_name_inf)]
    st.write('Nuevos casos confirmados: ',inf_data_ccaa[0])
    st.write('Nuevos fallecimientos: ',inf_data_ccaa[1])
    st.write('Nuevos casos hospitalizados: ',inf_data_ccaa[2])
    st.write('Nuevos ingresados en UCI: ',int(inf_data_ccaa[3]))

    st.write('Nota: Los hospitalizados empezaron a informarse a partir del 20 de marzo de 2020.')

if section_ind=='Evolución por género':
    st.title('Evolución por género')
    data_sexage = load_data_sexage()

    data_fem_last = data_sexage.query('dia=="{}"&gender=="Fem"'.format(data_sexage["dia"].max()))
    data_masc_last = data_sexage.query('dia=="{}"&gender=="Masc"'.format(data_sexage["dia"].max()))

    x = np.arange(len(data_fem_last['edad']))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12,9))
    rects1 = ax.bar(x - width/2, data_fem_last['total_casos'], width, color='purple', label='Mujeres')
    rects2 = ax.bar(x + width/2, data_masc_last['total_casos'], width, color='green', label='Hombres')

    st.markdown('## Infectados acumulados a día {}'.format(data_sexage["dia"].max().strftime('%Y-%m-%d')))
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Grupos de edad')
    ax.set_ylabel('Casos infectados')
    ax.set_title('Casos infectados por género y grupo de edad')
    ax.set_xticks(x)
    ax.set_xticklabels(data_fem_last['edad'])
    ax.legend()
    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    st.pyplot()

    st.markdown('## Evolución de la letalidad para mujeres')
    plot_letality_gender('Fem')

    st.markdown('## Evolución de la letalidad para hombres')
    plot_letality_gender('Masc')

if section_ind=='Series temporales: Estudio a corto plazo':
    st.title('Series temporales: Estudio a corto plazo')
    st.markdown('''
    En esta sección se presentan: el ajuste sobre la tendencia de las tasas de
    variación; así como una comparación entre las predicciones realizadas
    cada uno de los días previos y una tabla numérica con las diferentes predicciones
    realizadas en días previos, utilizando la misma metodología que en la fase de
    validación, no de calibración, y algunos de los errores estimados.
    ''')
    data, rel, ccaa_dict, n_prov, pob, data_pob = load_data()
    data_pob_agg = data_pob[['dia', 'total_casos', 'deaths', 'hospit', 'ingr_UCI', 'pob']].groupby('dia').agg('sum').reset_index()
    d = list(data_pob_agg['deaths'])
    post_days = 5
    geom_days = 5
    day_ini = data_pob_agg['dia'].min()

    d_long_0, preds_fut_long_0, days_long = predict_geoserie(d, 0, post_days, geom_days, day_ini)
    d_long_1, preds_fut_long_1, _ = predict_geoserie(d, 1, post_days+1, geom_days, day_ini)
    d_long_2, preds_fut_long_2, _ = predict_geoserie(d, 2, post_days+2, geom_days, day_ini)
    d_long_3, preds_fut_long_3, _ = predict_geoserie(d, 3, post_days+3, geom_days, day_ini)

    days_long_str = [d.strftime('%Y-%m-%d') for d in days_long]

    fig = plt.figure(facecolor='w',figsize=(12,9))
    ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(days_long_str, d_long_0, 'black', alpha=0.5, lw=3, label='Observado hasta el {}'.format(days_long[(-post_days-1)].strftime('%Y-%m-%d')))
    ax.plot(days_long_str, preds_fut_long_0, 'o-r', alpha=0.5, lw=2, label='Predicción con datos hasta el {}'.format(days_long[(-post_days-1)].strftime('%Y-%m-%d')))
    ax.plot(days_long_str, preds_fut_long_1, 'o-b', alpha=0.5, lw=2, label='Predicción con datos hasta el {}'.format(days_long[(-1-post_days-1)].strftime('%Y-%m-%d')))
    ax.plot(days_long_str, preds_fut_long_2, 'o-y', alpha=0.5, lw=2, label='Predicción con datos hasta el {}'.format(days_long[(-2-post_days-1)].strftime('%Y-%m-%d')))
    ax.plot(days_long_str, preds_fut_long_3, 'o-g', alpha=0.5, lw=2, label='Predicción con datos hasta el {}'.format(days_long[(-3-post_days-1)].strftime('%Y-%m-%d')))
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

    results = pd.DataFrame({'Día': [d_f.strftime('%Y-%m-%d') for d_f in days_long],
                            'Observado': d_long_0,
                            'Pred_3':[int(x) if x is not None else None for x in preds_fut_long_3],
                            'Pred_2':[int(x) if x is not None else None for x in preds_fut_long_2],
                            'Pred_1':[int(x) if x is not None else None for x in preds_fut_long_1],
                            'Pred_0':[int(x) if x is not None else None for x in preds_fut_long_0]})

    results_fmt = pd.DataFrame({'Día': [d_f.strftime('%Y-%m-%d') for d_f in days_long],
                            'Observado': remove_na(d_long_0),
                            'Pred_3':remove_na([int(x) if x is not None else None for x in preds_fut_long_3]),
                            'Pred_2':remove_na([int(x) if x is not None else None for x in preds_fut_long_2]),
                            'Pred_1':remove_na([int(x) if x is not None else None for x in preds_fut_long_1]),
                            'Pred_0':remove_na([int(x) if x is not None else None for x in preds_fut_long_0])})
    st.table(results_fmt.style.applymap(color_red,subset=['Pred_0','Pred_1','Pred_2','Pred_3']))

    diffs = pd.DataFrame({'Día':results['Día'],
                        'Observado': d_long_0,
                        'Pred_3_error':(results['Observado']-results['Pred_3'])/results['Pred_3'],
                        'Pred_2_error':(results['Observado']-results['Pred_2'])/results['Pred_2'],
                        'Pred_1_error':(results['Observado']-results['Pred_1'])/results['Pred_1']})
    
    st.table(diffs[(-post_days-3):-post_days].reset_index().drop('index',axis=1).style.applymap(color_red,subset=['Pred_1_error','Pred_2_error','Pred_3_error']).format({'Observado':"{:.0f}",
                        'Pred_3_error':"{:.2%}",
                        'Pred_2_error':"{:.2%}",
                        'Pred_1_error':"{:.2%}"}, na_rep=""))
    st.markdown('## Con suavizado')
    results_smt_aux = pd.DataFrame({'Día': [d_f.strftime('%Y-%m-%d') for d_f in days_long],
                        'Observado': d_long_0,
                        'Pred_3':[int(x) if x is not None else None for x in preds_fut_long_3],
                        'Pred_2_smooth':(results['Pred_3']+results['Pred_2'])/2,
                        'Pred_1_smooth':(results['Pred_3']+results['Pred_2']+results['Pred_1'])/3,
                        'Pred_0_smooth':(results['Pred_3']+results['Pred_2']+results['Pred_1']+results['Pred_0'])/4})
    results_smt = results_smt_aux.where(pd.notnull(results_smt_aux), None)

    results_fmt_smt = pd.DataFrame({'Día': [d_f.strftime('%Y-%m-%d') for d_f in days_long],
                            'Observado': remove_na(d_long_0),
                            'Pred_3':remove_na([int(x) if x is not None else None for x in preds_fut_long_3]),
                            'Pred_2_smooth':remove_na([int(x) if x is not None else None for x in results_smt['Pred_2_smooth']]),
                            'Pred_1_smooth':remove_na([int(x) if x is not None else None for x in results_smt['Pred_1_smooth']]),
                            'Pred_0_smooth':remove_na([int(x) if x is not None else None for x in results_smt['Pred_0_smooth']])})

    st.table(results_fmt_smt.style.applymap(color_red,subset=['Pred_0_smooth','Pred_1_smooth','Pred_2_smooth','Pred_3']))

    diffs_smt = pd.DataFrame({'Día':results_smt['Día'],
                        'Observado': d_long_0,
                        'Pred_3_smooth_error':(results_smt['Observado']-results_smt['Pred_3'])/results_smt['Pred_3'],
                        'Pred_2_smooth_error':(results_smt['Observado']-results_smt['Pred_2_smooth'])/results_smt['Pred_2_smooth'],
                        'Pred_1_smooth_error':(results_smt['Observado']-results_smt['Pred_1_smooth'])/results_smt['Pred_1_smooth']})

    st.table(diffs_smt[(-post_days-3):-post_days].reset_index().drop('index',axis=1).style.applymap(color_red,subset=['Pred_1_smooth_error','Pred_2_smooth_error','Pred_3_smooth_error']).format({'Observado':"{:.0f}",
                        'Pred_3_smooth_error':"{:.2%}",
                        'Pred_2_smooth_error':"{:.2%}",
                        'Pred_1_smooth_error':"{:.2%}"}, na_rep=""))

if section_ind=='GLM: Estudio a corto plazo':
    st.title('Modelos GLM: Estudio a corto plazo')
    st.write('''En este apartado se estudia la evolución a corto plazo a nivel de CCAA según un modelo GLM Poisson con el día y la
    comunidad autónoma como únicos factores de riesgo. Hay que tener en cuenta que este modelo predecirá bien los próximos días pero
    no lo hará para un largo plazo, ya que en ningún momento se tiene en cuenta una futura bajada del contagio.
    ''')
    data, rel, ccaa_dict, n_prov, _, _ = load_data()
    responses_display = {'deaths':'Fallecimientos', 'total_casos':'Casos observados'}
    response = st.selectbox('Respuesta',['deaths', 'total_casos'],format_func=lambda x: responses_display[x])
    predict_df_full, response_df, days_fut_fmt = train_glm(response)

    spain_select = st.checkbox('Predicción estatal',['España','CCAA'],key='spain_select_glm')

    if spain_select:
        st.write('Nota: Ceuta y Melilla no se incluyen en el estudio.')
        ca_select = 'Spain'
        predict_fut = [predict_df_full[i].sum(1) for i in range(len(predict_df_full))]
        response_fut = pd.DataFrame(response_df).sum(1)
        response_fut[-3:] = None
    else:
        ca_select = st.selectbox('CCAA',list(ccaa_dict.keys()),index=12,key='ca_name_glm')
        predict_fut = [pred_ca[ca_select] for pred_ca in predict_df_full]
        response_fut = response_df[ca_select]

    cmap = cm.get_cmap('winter',len(predict_fut))
    newcolors = cmap(np.linspace(0, 1, len(predict_fut)))

    fig = plt.figure(facecolor='w',figsize=(12,9))
    ax = fig.add_subplot(111, axisbelow=True)

    for i in range(len(predict_fut)):
        ax.plot(days_fut_fmt, predict_fut[i], 'b', alpha=0.5, lw=2, label='Predicted, until {}'.format(days_fut_fmt[-4-i]),color=newcolors[i])
    ax.plot(days_fut_fmt, response_fut, 'r', alpha=0.5, lw=3, label='Observed')

    ax.set_xlabel('Día')
    ax.set_ylabel(response)
    if spain_select:
        plt.ylim((0,response_fut.max()*2))
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0.1,rotation=45)
    plt.xticks(ha='right')
    ax.grid(b=True, which='major', c='grey', lw=0.5, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(1)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.title('{}: Prediction of the evolution of the {}, COVID-19'.format(ca_select,response))
    st.pyplot()

    st.table(pd.DataFrame({'Day':days_fut_fmt,
                'Observed':remove_na(response_fut),
                'Pred_unt_{}'.format(days_fut_fmt[-4-4]):remove_na(extract_only_fut_preds(4)),
                'Pred_unt_{}'.format(days_fut_fmt[-4-3]):remove_na(extract_only_fut_preds(3)),
                'Pred_unt_{}'.format(days_fut_fmt[-4-2]):remove_na(extract_only_fut_preds(2)),
                'Pred_unt_{}'.format(days_fut_fmt[-4-1]):remove_na(extract_only_fut_preds(1)),
                'Pred_unt_{}'.format(days_fut_fmt[-4-0]):remove_na(extract_only_fut_preds(0))})\
    .style.applymap(color_red, subset=['Pred_unt_{}'.format(days_fut_fmt[-1-i]) for i in range(3,8)]))

    error = []

    for i in range(1,5):
        error.append(list(np.repeat(None,4-i)) + [(p-r)/r for r,p in zip(response_fut[(-3-i):-3],extract_only_fut_preds(i)[(-3-i):-3])])

    diffs = pd.DataFrame({'Día':days_fut_fmt[-7:-3],
                        'Observado': remove_na(response_fut[-7:-3]),
                        'Error_until_{}'.format(days_fut_fmt[-4-4]):remove_na(error[3],pct=True),
                        'Error_until_{}'.format(days_fut_fmt[-4-3]):remove_na(error[2],pct=True),
                        'Error_until_{}'.format(days_fut_fmt[-4-2]):remove_na(error[1],pct=True),
                        'Error_until_{}'.format(days_fut_fmt[-4-1]):remove_na(error[0],pct=True)})\
    .style.applymap(color_red, subset=['Error_until_{}'.format(days_fut_fmt[-4-i]) for i in range(1,5)])

    st.table(diffs)
    

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
        data_pob_agg = data_pob[['dia', 'total_casos', 'deaths', 'hospit', 'ingr_UCI', 'pob']].groupby('dia').agg('sum').reset_index()
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
        ca_name_long = st.selectbox('CCAA',list(ccaa_dict.keys()),index=12,key='long')
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
    ax.grid(b=True, which='major', c='grey', lw=0.5, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.title('{}: Prediction of the evolution of the COVID-19'.format(ca_name_long))
    st.pyplot()

    detail_lim = len(list(data_pob['dia'].unique()))
        # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w',figsize=(12,9))
    ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(days_str, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(days_str, I/N, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(days_str, R/N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.plot(days_str,observed, 'black', alpha=1, lw=3, label='Observed Infected')
    ax.set_xlabel('Time/days')
    ax.set_ylabel('% People')
    ax.set_ylim(0,((I/N)[:detail_lim]).max()*1.1)
    #ax.set_ylim(0,0.001)
    ax.set_xlim(0,detail_lim)
    loc = plticker.MultipleLocator(base=3) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=2,rotation=45)
    plt.xticks(ha='right')
    ax.grid(b=True, which='major', c='grey', lw=0.5, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.title('{}: Detail of prediction of the evolution of the COVID-19'.format(ca_name_long))
    st.pyplot()

    val_inf, idx_inf = max((val, idx) for (idx, val) in enumerate(I))
    st.write('Se espera que en {}, el {} se alcance el pico máximo de infectados con un {:.2%} de la población afectada.'.format(ca_name_long,days_str[idx_inf],val_inf/N))
    st.write('''Nota: Este modelo SIR representa el escenario que se esperaría sin tomar ninguna medida de contención. 
    En la actualidad, estamos trabajando en el modelo SEIR para ampliar los resultados.
    ''')

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