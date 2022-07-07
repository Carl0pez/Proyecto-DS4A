import pandas as pd
import matplotlib.pylab as plt
import pickle
from Dash.app_dataframe import df_hom


def sarimax_forecast_city(SARIMAX_model, df):

    fcast = SARIMAX_model.get_prediction(start=1, end=len(df['cantidad']))
    ts_p = fcast.predicted_mean
    ts_ci = fcast.conf_int()

    #Plot results
    plt.figure(figsize=(22,6))
    plt.plot(ts_p,label='prediction')
    plt.plot(df['cantidad'],color='red',label='actual')
    plt.fill_between(ts_ci.index[1:],
                    ts_ci.iloc[1:, 0],
                    ts_ci.iloc[1:, 1], color='k', alpha=.2)
    plt.title('Homicides per date')
    plt.xlabel('Date')
    plt.ylabel('Total of homicides')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()
    


city = 'cal'

if city == 'med':
    df = df_hom[df_hom['municipio']=='MEDELLÍN (CT)'][['fecha','cantidad']]
elif city == 'bog':
    df = df_hom[df_hom['municipio']=='BOGOTÁ D.C. (CT)'][['fecha','cantidad']]
elif city == 'cal':
    df = df_hom[df_hom['municipio']=='CALI (CT)'][['fecha','cantidad']]
else:
    df = df_hom[['fecha','cantidad']]

df = df.set_index('fecha')
df = df.resample('M').sum()
df['month_index'] = df.index.month

if city == 'med':
    name = 'SARIMAX_med.pkl'
elif city == 'bog':
    name = 'SARIMAX_bog.pkl'
elif city == 'cal':
    name = 'SARIMAX_cal.pkl'
else:
    name = 'SARIMAX_total.pkl'

with open(name, 'rb') as pkl:
    SARIMA_model = pickle.load(pkl)
    sarimax_forecast_city(SARIMA_model,df)
