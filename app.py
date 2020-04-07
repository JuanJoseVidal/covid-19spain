import streamlit as st
import pandas as pd
import numpy as np
import datetime
import math
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
    return list(np.repeat(None,len(predict_fut[i])-7-i)) + [int(p) for p in predict_fut[i][(-7-i):]]

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def predict_model(res,ca,day):
    pred = [1] + list(np.repeat(0,16)) + [1]
    if ca == 'Madrid':
        pred = pred + [1]
    else:
        pred = pred + [0]
    pred = pred + [day**2] + [day**3] + [day]
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
            y, X = dmatrices(f'{response} ~ I(day**2) + I(day**3) + C(CCAA)*day + confin + C(madrid)', data=data_red, return_type='dataframe')

            mod = sm.GLM(y, X, family=sm.families.Poisson(), link=sm.families.links.logit)
            res.append(mod.fit())
            data_plot = data_red.query(f'CCAA_Name=="{ca}"')[['day',response]]
            if i==0:
                response_df.update({ca:list(data_plot[response]) + list(np.repeat(None,7+i))})
            days_fut = list(data_plot['day']) + list(range(data_plot['day'].max()+1,data_plot['day'].max()+8+i))

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

    for i in range(len(predict_df_full)):
        predict_df_full[i] = predict_df_full[i].applymap(lambda x: x[0])
        for ca in list(ccaa_dict.keys()):
            predict_df_full[i][ca].loc[predict_df_full[i][ca].idxmax():] = predict_df_full[i][ca].max()

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

def Runge4D(f1,f2,f3,f4,h,t0,T,x0,y0,z0,v0,beta,beta1,gamma,sigma,alfa,N):
    
    N1 = math.ceil((T-t0)/h)
    x = np.repeat(None, N1+1)
    y = np.repeat(None, N1+1)
    z = np.repeat(None, N1+1)
    v = np.repeat(None, N1+1)
    t = np.linspace(t0,T,math.ceil(N1+1))
    x[0] = x0
    y[0] = y0
    z[0] = z0
    v[0] = v0

    for i in range(N1):
        r1=f1(t[i],x[i],y[i],z[i],v[i],beta,beta1,gamma,sigma,alfa,N)
        u1=f2(t[i],x[i],y[i],z[i],v[i],beta,beta1,gamma,sigma,alfa,N)
        v1=f3(t[i],x[i],y[i],z[i],v[i],beta,beta1,gamma,sigma,alfa,N)
        d1=f4(t[i],x[i],y[i],z[i],v[i],beta,beta1,gamma,sigma,alfa,N)

        z1=x[i]+h*r1/2
        w1=y[i]+h*u1/2
        p1=z[i]+h*v1/2
        g1=v[i]+h*d1/2

        r2=f1(t[i]+h/2,z1,w1,p1,g1,beta,beta1,gamma,sigma,alfa,N)
        u2=f2(t[i]+h/2,z1,w1,p1,g1,beta,beta1,gamma,sigma,alfa,N)
        v2=f3(t[i]+h/2,z1,w1,p1,g1,beta,beta1,gamma,sigma,alfa,N)
        d2=f4(t[i]+h/2,z1,w1,p1,g1,beta,beta1,gamma,sigma,alfa,N)

        z2=x[i]+h*r2/2
        w2=y[i]+h*u2/2
        p2=z[i]+h*v2/2
        g2=v[i]+h*d2/2
                    
        r3=f1(t[i]+h/2,z2,w2,p2,g2,beta,beta1,gamma,sigma,alfa,N)
        u3=f2(t[i]+h/2,z2,w2,p2,g2,beta,beta1,gamma,sigma,alfa,N)
        v3=f3(t[i]+h/2,z2,w2,p2,g2,beta,beta1,gamma,sigma,alfa,N)
        d3=f4(t[i]+h/2,z2,w2,p2,g2,beta,beta1,gamma,sigma,alfa,N)

        z3=x[i]+h*r3
        w3=y[i]+h*u3
        p3=z[i]+h*v3
        g3=v[i]+h*d3
                    
        r4=f1(t[i]+h,z3,w3,p3,g3,beta,beta1,gamma,sigma,alfa,N)
        u4=f2(t[i]+h,z3,w3,p3,g3,beta,beta1,gamma,sigma,alfa,N)
        v4=f3(t[i]+h,z3,w3,p3,g3,beta,beta1,gamma,sigma,alfa,N)
        d4=f4(t[i]+h,z3,w3,p3,g3,beta,beta1,gamma,sigma,alfa,N)

        x[i+1]=x[i]+h*(r1+2*r2+2*r3+r4)/6
        y[i+1]=y[i]+h*(u1+2*u2+2*u3+u4)/6
        z[i+1]=z[i]+h*(v1+2*v2+2*v3+v4)/6
        v[i+1]=v[i]+h*(d1+2*d2+2*d3+d4)/6

    return t, x, y, z, v

def f1(t,x,y,z,v,beta,beta1,gamma,sigma,alfa,N):
    return -(beta-beta1*(1-np.exp(-alfa*(t-22))))*(1-0.1*(0*y+z)/N)**200*x*(y+z)/N
def f2(t,x,y,z,v,beta,beta1,gamma,sigma,alfa,N):
    return (beta-beta1*(1-np.exp(-alfa*(t-22))))*(1-0.1*(0*y+z)/N)**200*x*(y+z)/N-sigma*y
def f3(t,x,y,z,v,beta,beta1,gamma,sigma,alfa,N):
    return sigma*y-gamma*z
def f4(t,x,y,z,v,beta,beta1,gamma,sigma,alfa,N):
    return gamma*z

def SEIRRest(N,E0,I0,R0,beta,beta1,gamma,sigma,T,alfa):
    
    S0=N-I0-R0-E0

    t,S,E,I,R = Runge4D(f1,f2,f3,f4,0.05,22,T,S0,E0,I0,R0,beta,beta1,gamma,sigma,alfa,N)
    return t,S,E,I,R

def extract_time(vect):
    vect_time = []
    for i in range(len(vect)):
        if i%20==0:
            vect_time.append(vect[i])
        else:
            pass
    return vect_time

st.sidebar.title('Índice')
section_ind = st.sidebar.radio('',['Introducción', 'Informe diario', 'Evolución por género', 'Series temporales: Estudio a corto plazo','GLM: Estudio a corto plazo','SEIR: Estudio a largo plazo','Documentación','Acerca del proyecto'])

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
    el número de infectados, el número de fallecidos, el número de recuperados y el número de ingresos en UCI, 
    para el total acumulado en el conjunto del territorio nacional como por comunidad autónoma.
    ''')    
    st.markdown('''
    ## Datos y Fuentes de Información
    Los datos utilizados son los publicados por el Gobierno Español en sus actualizaciones diarias, 
    aunque en la fase inicial se utilizaron los datos recopilados y depurados por el grupo de trabajo
    Datadista (Github) y los proporcionados por el Johns Hopkins CSSE.
    ''')

if section_ind=='Informe diario':
    st.title('Informe de evolución diaria.')
    data, rel, ccaa_dict, n_prov, pob, data_pob = load_data()
    data_pob_agg = data_pob[['dia', 'total_casos', 'deaths', 'hospit', 'ingr_UCI', 'curados', 'pob']].groupby('dia').agg('sum').reset_index()
    data_pob_agg['pct_death_pob'] = data_pob_agg['deaths']/data_pob_agg['pob']
    ult_day = data_pob_agg['dia'].max()
    st.markdown('## Acumulado a día {}.'.format(ult_day.strftime('%Y-%m-%d')))
    st.write('Casos confirmados: ',int(data_pob_agg.query(f'dia=="{ult_day}"').reset_index()['total_casos'][0]))
    st.write('Fallecimientos: ',int(data_pob_agg.query(f'dia=="{ult_day}"').reset_index()['deaths'][0]))
    st.write('Casos hospitalizados: ',int(data_pob_agg.query(f'dia=="{ult_day}"').reset_index()['hospit'][0]))
    st.write('Ingresados en UCI: ',int(data_pob_agg.query(f'dia=="{ult_day}"').reset_index()['ingr_UCI'][0]))
    st.write('Casos recuperados: ',int(data_pob_agg.query(f'dia=="{ult_day}"').reset_index()['curados'][0]))


    st.markdown('## Evolución diaria.')

    fig, ax1 = plt.subplots(figsize=(12,9))
    ax1.set_xlabel('Día')
    ax1.set_ylabel('Casos infectados (barras, diarios)', fontsize=20, color='grey')
    ax1.bar(data_pob_agg['dia'][1:], data_pob_agg['total_casos'].diff()[1:], color='grey',label='Casos infectados')
    ax1.tick_params(axis='y')
    ax1.bar(data_pob_agg['dia'][1:], data_pob_agg['curados'].diff()[1:], color='green', label='Curados')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Fallecimientos (línea, diarios)', fontsize=20, color='red')  # we already handled the x-label with ax1
    ax2.plot(data_pob_agg['dia'][1:], data_pob_agg['deaths'].diff()[1:], color='red', label='Fallecimientos')
    ax2.tick_params(axis='y')
    ax2.set_ylim(0)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    fig.tight_layout() 
    st.pyplot()

    dia_select = [d for d in data_pob_agg['dia'][1:]]
    day_format_func = lambda x: x.strftime('%Y-%m-%d') 
    dia_study = st.selectbox('Día',dia_select,key='day_informe',index=len(dia_select)-1,format_func=day_format_func)

    st.write('Nuevos casos confirmados: ',extract_diffs('total_casos', dia_study))
    st.write('Nuevos fallecimientos: ',extract_diffs('deaths', dia_study))
    st.write('Nuevos casos hospitalizados: ',extract_diffs('hospit', dia_study))
    st.write('Nuevos ingresados en UCI: ',extract_diffs('ingr_UCI', dia_study))
    st.write('Nuevos casos recuperados: ',extract_diffs('curados', dia_study))

    ca_name_inf = st.selectbox('CCAA',list(ccaa_dict.keys()),index=12,key='ca_informe')
    inf_data_ccaa = [extract_diffs_ccaa('total_casos', dia_study, ca_name_inf),\
    extract_diffs_ccaa('deaths', dia_study, ca_name_inf),\
    extract_diffs_ccaa('hospit', dia_study, ca_name_inf),\
    extract_diffs_ccaa('ingr_UCI', dia_study, ca_name_inf),\
    extract_diffs_ccaa('curados', dia_study, ca_name_inf)]
    st.write('Nuevos casos confirmados: ',inf_data_ccaa[0])
    st.write('Nuevos fallecimientos: ',inf_data_ccaa[1])
    st.write('Nuevos casos hospitalizados: ',inf_data_ccaa[2])
    st.write('Nuevos ingresados en UCI: ',int(inf_data_ccaa[3]))
    st.write('Nuevos casos recuperados: ',int(inf_data_ccaa[4]))
    

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
    data_pob_agg = data_pob[['dia', 'total_casos', 'deaths', 'hospit', 'ingr_UCI', 'curados','pob']].groupby('dia').agg('sum').reset_index()
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
    st.write('Nota: El error se calcula de la forma siguiente:')
    st.latex(r'Error = \frac{Predicción-Observado}{Observado}')

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
        response_fut[-7:] = None
    else:
        ca_select = st.selectbox('CCAA',list(ccaa_dict.keys()),index=12,key='ca_name_glm')
        predict_fut = [pred_ca[ca_select] for pred_ca in predict_df_full]
        response_fut = response_df[ca_select]

    cmap = cm.get_cmap('winter',len(predict_fut))
    newcolors = cmap(np.linspace(0, 1, len(predict_fut)))

    fig = plt.figure(facecolor='w',figsize=(12,9))
    ax = fig.add_subplot(111, axisbelow=True)

    for i in range(len(predict_fut)):
        ax.plot(days_fut_fmt, predict_fut[i], 'b', alpha=0.5, lw=2, label='Predicted, until {}'.format(days_fut_fmt[-8-i]),color=newcolors[i])
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
                'Pred_unt_{}'.format(days_fut_fmt[-8-4]):remove_na(extract_only_fut_preds(4)),
                'Pred_unt_{}'.format(days_fut_fmt[-8-3]):remove_na(extract_only_fut_preds(3)),
                'Pred_unt_{}'.format(days_fut_fmt[-8-2]):remove_na(extract_only_fut_preds(2)),
                'Pred_unt_{}'.format(days_fut_fmt[-8-1]):remove_na(extract_only_fut_preds(1)),
                'Pred_unt_{}'.format(days_fut_fmt[-8-0]):remove_na(extract_only_fut_preds(0))})\
    .style.applymap(color_red, subset=['Pred_unt_{}'.format(days_fut_fmt[-1-i]) for i in range(7,12)]))

    error = []

    for i in range(1,5):
        error.append(list(np.repeat(None,4-i)) + [(p-r)/r for r,p in zip(response_fut[(-7-i):-7],extract_only_fut_preds(i)[(-7-i):-7])])

    diffs = pd.DataFrame({'Día':days_fut_fmt[-11:-7],
                        'Observado': remove_na(response_fut[-11:-7]),
                        'Error_until_{}'.format(days_fut_fmt[-8-4]):remove_na(error[3],pct=True),
                        'Error_until_{}'.format(days_fut_fmt[-8-3]):remove_na(error[2],pct=True),
                        'Error_until_{}'.format(days_fut_fmt[-8-2]):remove_na(error[1],pct=True),
                        'Error_until_{}'.format(days_fut_fmt[-8-1]):remove_na(error[0],pct=True)})\
    .style.applymap(color_red, subset=['Error_until_{}'.format(days_fut_fmt[-4-i]) for i in range(5,9)])

    st.table(diffs)
    st.write('Nota: El error se calcula de la forma siguiente:')
    st.latex(r'Error = \frac{Predicción-Observado}{Observado}')


if section_ind=='SEIR: Estudio a largo plazo':
    st.title('Modelos SEIR: Estudio a largo plazo')
    st.write('''En este apartado se estudia la evolución a largo plazo a nivel de país. Previamente se han ajustado los parámetros 
    correspondientes al modelo SEIR explicado en el apartado Documentación. Este modelo intenta predecir el pico de casos a largo 
    terminio teniendo en cuenta una recuperación e inmunidad posterior por la sociedad, al igual que una limitación de personas expuestas
    al momento al virus y el impacto de las medidas tomadas por el gobierno.
    ''')
    
    data, rel, ccaa_dict, n_prov, pob, data_pob = load_data()
    data_pob_agg = data_pob[['dia', 'total_casos', 'deaths', 'hospit', 'ingr_UCI', 'curados', 'pob']].groupby('dia').agg('sum').reset_index()
    data_pob_agg['pct_death_pob'] = data_pob_agg['deaths']/data_pob_agg['pob']
    
    data_dia = data_pob_agg[(data_pob_agg['dia'] == '2020-03-21')].reset_index(drop=True)
    data_study = data_pob_agg[(data_pob_agg['dia'] >= '2020-03-21')].reset_index(drop=True)
    data_dia['curados'] = 2575 # El dato para el 21 de marzo está en el informe pero agregado y no por CCAA
    
    T = 60

    alfa = 0.16
    gamma = 0.06
    beta = 0.16
    beta_1 = 0.13

    t,S,E,I,R = SEIRRest(N = data_dia['pob'][0],
                        E0 = 40000,
                        I0 = data_dia['total_casos'][0] - data_dia['deaths'][0] - data_dia['curados'][0],
                        R0 = data_dia['deaths'][0] + data_dia['curados'][0],
                        beta = beta,
                        beta1 = beta_1,
                        gamma = gamma,
                        sigma = 1/7,
                        T = T,
                        alfa = alfa)

    max_days = len(extract_time(t))
    days = [data_dia['dia'][0]]
    for i in range(max_days-1): 
            days.append(days[i] + datetime.timedelta(days=1))
    days_fut_fmt = [d.strftime('%Y-%m-%d') for d in days]

    observed = list(data_study['total_casos']-data_study['deaths']-data_study['curados']) + list(np.repeat(None,len(extract_time(t))-len(data_study['total_casos'])))

    rd = [r+d for r,d in zip(data_study['deaths'],data_study['curados'])]
    rd = rd + list(np.repeat(None,len(extract_time(t))-len(rd)))

    st.write('Los parámetros ajustados del modelo son los siguientes:')
    st.latex(r'\alpha = {}\ \ \ \gamma = {}'.format(alfa,gamma))
    st.latex(r'\beta = {}\ \ \ \beta_1 = {}'.format(beta,beta_1))

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w',figsize=(12,9))
    ax = fig.add_subplot(111, axisbelow=True)
    #ax.plot(t, S/pob, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(days_fut_fmt, extract_time(I), 'lightcoral', alpha=0.5, lw=3, label='Infected')
    ax.plot(days_fut_fmt, extract_time(R), 'lightgreen', alpha=0.5, lw=3, label='Recovered/Death')
    ax.plot(days_fut_fmt, extract_time(E), 'khaki', alpha=0.5, lw=3, label='Exposed')
    ax.plot(days_fut_fmt, observed, 'darkred', alpha=0.5, lw=3, label='Infected (observed)')
    ax.plot(days_fut_fmt, rd, 'darkgreen', alpha=0.5, lw=3, label='Recovered/Death (observed)')
    ax.set_xlabel('Time/days')
    ax.set_ylabel('People')
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=2,rotation=45)
    plt.xticks(ha='right')
    loc = plticker.MultipleLocator(base=3) # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)
    legend = ax.legend()
    st.pyplot()

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
    st.markdown('''**José Valero Cuadra**  
    CIO en la Universitat Miguel Hernández.  
    Mail: [jvalero@umh.es](mailto:jvalero@umh.es)  
    ''')