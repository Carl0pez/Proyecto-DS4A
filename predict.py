import pandas as pd
import matplotlib.pylab as plt
import pickle
from Dash.app_dataframe import df_hom


def sarimax_forecast_city(SARIMAX_model, df, periods):
    # Forecast
    n_periods = periods

    forecast_df = pd.DataFrame({"month_index":pd.date_range(df.index[-1], periods = n_periods, freq='MS').month},
                    index = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods = n_periods, freq='MS'))

    fitted, confint = SARIMAX_model.predict(n_periods=n_periods, 
                                            return_conf_int=True,
                                            exogenous=forecast_df[['month_index']])
    index_of_fc = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods = n_periods, freq='MS')

    # make series for plotting purpose
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.figure(figsize=(15,7))
    plt.plot(df["total"], color='#1f76b4')
    plt.plot(fitted_series, color='darkgreen')
    plt.fill_between(lower_series.index, 
                    lower_series, 
                    upper_series, 
                    color='k', alpha=.15)

    plt.title("SARIMAX - Forecast of Airline Passengers")
    plt.show()

city = 'med'

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
df.columns = ['total']
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
    sarimax_forecast_city(SARIMA_model,df, 5)
