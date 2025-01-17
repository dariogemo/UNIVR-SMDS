import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.signal import periodogram, find_peaks
import matplotlib.pyplot as plt
from seaborn import lineplot
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import jarque_bera
from sklearn.preprocessing import StandardScaler
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def preprocess_df(df):
    all_nation_list = df.index.unique()
    all_val_nation_list = []
    for nation in all_nation_list:
        if len(df[df.index == nation].T.reset_index().iloc[1]) == 18:
            all_val_nation_list.append(nation)
        else:
            pass

    all_val_nation_list

    df = df.loc[all_val_nation_list]
    count_df = df.groupby('Country').count()
    mask = count_df.sum(axis = 1) == 884
    valid_nations = list(count_df[mask].index)
    df = df[df['Country'].isin(valid_nations)]

    return df, valid_nations

def nation_input(number, valid_nations, df):   
    adj_list = ['first', 'second', 'third', 'fourth', 'fifth']
    while True:
        first = input(f'Please input {adj_list[number]} nation: ')
        first = first.lower().capitalize()
        number = number+1 
        if len(first) == 1:
            first = first.lower()
        if not first and number == 1:
            first = 'Finland'
        if not first and number == 2:
            first = 'Argentina'
        if not first and number == 3:
            first = 'Belgium'
        if not first and number == 4:
            first = 'Philippines'
        if not first and number == 5:
            first = 'Cuba'
        if first in valid_nations:
            break
        if first in df['Country'].unique() and first not in valid_nations:
            print(f'Not enough data for {first}.\nPlease enter a valid nation.\n-----------------')
        else:
            print(f'{first} is either misspelled or not a nation.\nPlease enter a valid nation.\n-----------------')
    return first

def create_nation_list(valid_nations, df): 
    nation_list = []
    for nation_num in range(5):
        nation = nation_input(nation_num, valid_nations, df)
        nation_list.append(nation)
    return nation_list

def create_df(df):
    df = df.reset_index().drop(['Country', 'CountryID'], axis = 1)
    df = df.T.reset_index().drop(['index'], axis = 1)
    df.columns = df.loc[0]
    df = df.drop(0)
    df.index = pd.date_range(start = 1970, periods = 51, freq = 'YE').year
    df = df[['Exports of goods and services', 'Imports of goods and services', 'Gross Domestic Product (GDP)', 'Manufacturing (ISIC D)', 'Gross capital formation']]
    df = df.iloc[:, [0, 1, 3, 4, 2]]
    df.columns = ['Exports', 'Imports', 'Manufacturing', 'Gross_capital', 'GDP']
    return df

def highest_corr_variable(corr_list, max_list, nation_list):
    for corr in corr_list:
        max_corr = corr['GDP'].drop('GDP').max()
        idx = corr.index[corr['GDP'] == max_corr].tolist()[0]
        max_list.append(idx)
    df = pd.DataFrame(max_list, index = nation_list, columns = ['Highest correlation variable'])
    display(df)

def lowest_corr_variable(corr_df):
    cgdpd = corr_df.drop('GDP', axis = 1).replace({1 : np.nan})
    idx = cgdpd.stack().idxmin()
    return idx

def train_test_split(df):
    train = df.loc[:2009]
    train = train.astype('float')
    test = df.loc[2010:]
    test = test.astype('float')
    lista = [train, test]
    return lista

def plotseasonal(res, axes):
    res.observed.plot(ax=axes[0], legend=False)
    axes[0].set_ylabel('Observed')
    res.trend.plot(ax=axes[1], legend=False)
    axes[1].set_ylabel('Trend')
    res.seasonal.plot(ax=axes[2], legend=False)
    axes[2].set_ylabel('Seasonal')
    res.resid.plot(ax=axes[3], legend=False)
    axes[3].set_ylabel('Residual')

def roll_mean_std_plot(df, nation_list, roll_num):
    fig, ax = plt.subplots(len(nation_list), 1, figsize = (15, 15))
    plt.suptitle('GDP of the nations', fontsize = 40)
    plt.tight_layout(pad = 2.5)

    for idx, nation in enumerate(nation_list):
        rolling_mean = df[nation][0]['GDP'].rolling(roll_num).mean()
        rolling_std = df[nation][0]['GDP'].rolling(roll_num).std()
        lineplot(df[nation][0]['GDP'], ax = ax[idx])
        ax[idx].plot(rolling_mean, color = 'red', label = 'Rolling Mean')
        ax[idx].plot(rolling_std, color = 'black', label = 'Rolling Std')
        ax[idx].set_title(nation, fontsize = 20)
        ax[idx].legend(loc = 'best')

    plt.show()

def check_stationarity(df_list, df_train_test, nation_list, dic = True):
    list_for_df = []
    if dic == True:
        for idx, df in enumerate(df_list):
            adf_test = adfuller(df_train_test[nation_list[idx]][0]['GDP'], autolag='AIC')
            lista_prova = []
            lista_prova.append(adf_test[0])
            lista_prova.append(adf_test[1])
            if adf_test[1] <= 0.05:
                lista_prova.append('Yes')
            else:
                lista_prova.append('No')
            kpss_out = kpss(df_train_test[nation_list[idx]][0]['GDP'], regression='c', nlags='auto', store=True)
            lista_prova.append(kpss_out[0])
            lista_prova.append(kpss_out[1])
            if kpss_out[1] >= 0.05:
                lista_prova.append('Yes')
            else:
                lista_prova.append('No')
            list_for_df.append(lista_prova)
        final_df = pd.DataFrame(list_for_df, 
                                columns = ['ADF', 'P-value for ADF', 'ADF stationarity', 'KPSS', 'P-value for KPSS', 'KPSS stationarity'], 
                                index = [nation_list])
    if dic == False:
        for idx, df in enumerate(df_list):
            adf_test = adfuller(df_train_test[nation_list[idx]], autolag='AIC')
            lista_prova = []
            lista_prova.append(adf_test[0])
            lista_prova.append(adf_test[1])
            if adf_test[1] <= 0.05:
                lista_prova.append('Yes')
            else:
                lista_prova.append('No')
            kpss_out = kpss(df_train_test[nation_list[idx]], regression='c', nlags='auto', store=True)
            lista_prova.append(kpss_out[0])
            lista_prova.append(kpss_out[1])
            if kpss_out[1] >= 0.05:
                lista_prova.append('Yes')
            else:
                lista_prova.append('No')
            list_for_df.append(lista_prova)
        final_df = pd.DataFrame(list_for_df, 
                                columns = ['ADF', 'P-value for ADF', 'ADF stationarity', 'KPSS', 'P-value for KPSS', 'KPSS stationarity'], 
                                index = [nation_list])
    return final_df

def log_transform(df_train_test):
    df_train_test_x = {}
    for nation in df_train_test:
        df_train_test_x[nation] = (np.log(df_train_test[nation][0]), np.log(df_train_test[nation][1]))
    return df_train_test_x

def difference(df_train_test, difference, nation):
    tuple = (df_train_test[nation][0].diff(difference).dropna(), df_train_test[nation][1].diff().dropna())
    return tuple

def detrend(df_train_test_log_dif, nation_list, seasons_list):
    df_train_test_log_dif_detrend = {}
    for idx, nation in enumerate(nation_list):
        stl = STL(df_train_test_log_dif[nation][0]['GDP'], period = seasons_list[idx])
        result = stl.fit()
        trend = result.trend
        df_train_test_log_dif_detrend[nation] = df_train_test_log_dif[nation][0]['GDP'] - trend
    return df_train_test_log_dif_detrend

def spd(nation, df_train_test, Fs):
    f_per, Pxx_per = periodogram(df_train_test[nation][0]['GDP'], Fs, detrend = None,window = 'triang',return_onesided = True, scaling = 'density')
    f_per = f_per[1:]
    Pxx_per = Pxx_per[1:]

    #Find the peaks of the periodogram.
    peaks = find_peaks(Pxx_per[f_per >= 0], prominence = 100000)[0]
    peak_freq = f_per[peaks]
    peak_dens = Pxx_per[peaks]

    #Plot of the analysis transformation and of its peaks. Only the first five are interestings
    plt.figure(figsize=(16,6))
    plt.title('Periodogram of the GDP of ' + nation)

    plt.plot(peak_freq[:5], peak_dens[:5], 'ro');
    plt.plot(f_per[2:],Pxx_per[2:])

    #Retrieving of the values
    data = {'Frequency': peak_freq, 'Density': peak_dens, 'Period': 1/peak_freq}
    period_df = pd.DataFrame(data)
    season_list = 1/peak_freq
    top_season = season_list[0]

    plt.plot(f_per, Pxx_per)
    plt.plot(f_per, Pxx_per)
    plt.xlabel('Sample Frequencies')
    plt.ylabel('Power')
    plt.show()

    display(period_df.head(5))

    return top_season

# PREDICTIONS

def create_metrics_df():
    df_metrics_1 = pd.DataFrame(columns = ['Model_name','AIC','MAE','RMSE','MAPE'])
    df_metrics_2 = pd.DataFrame(columns = ['Model_name','AIC','MAE','RMSE','MAPE'])
    df_metrics_3 = pd.DataFrame(columns = ['Model_name','AIC','MAE','RMSE','MAPE'])
    df_metrics_4 = pd.DataFrame(columns = ['Model_name','AIC','MAE','RMSE','MAPE'])
    df_metrics_5 = pd.DataFrame(columns = ['Model_name','AIC','MAE','RMSE','MAPE'])
    metrics_list_of_df = [df_metrics_1, df_metrics_2, df_metrics_3, df_metrics_4, df_metrics_5]
    return metrics_list_of_df

def arima_order(nation_list, df_train_test):
    order_list = []
    arima_model_list = []
    for idx, nation in enumerate(nation_list):
        scaler = StandardScaler()
        print(nation)
        X_train_scaled = scaler.fit_transform(df_train_test[nation][0].drop('GDP', axis=1))
        X_test_scaled = scaler.transform(df_train_test[nation][1].drop('GDP', axis=1))
        GDP_train_scaled = scaler.fit_transform(pd.DataFrame(df_train_test[nation][0]['GDP']))
        GDP_test_scaled = scaler.fit_transform(pd.DataFrame(df_train_test[nation][1]['GDP']))

        best_model = auto_arima(
        GDP_train_scaled, 
        X = X_train_scaled,
        start_p = 0, d = 2, start_q = 0, 
        max_p = 5, max_q = 5,
        seasonal = False,
        error_action = 'warn', 
        with_intercept = True, 
        trace = True, 
        suppress_warnings = True,
        stepwise = True,
        random_state = 20, 
        information_criterion = 'aic'
        )

        order_list.append(best_model.order)

        arimax_model = ARIMA(GDP_train_scaled, 
                     exog = X_train_scaled,
                     order = best_model.order
                     ).fit()
        
        arima_model_list.append(arimax_model)
    return order_list, arima_model_list

def arima_diagnostics(arima_model_list, nation_list):
    for idx, model in enumerate(arima_model_list):
        print(f"Summary and diagnostics for {nation_list[idx]}'s arima model\n")
        print(model.summary())
        model.plot_diagnostics(figsize = (10, 7))
        plt.show()
        print('--------------------------------------')

def arima_res_stats(arima_model_list, nation_list, df_train_test):
    for idx, model in enumerate(arima_model_list):
        stand_resid = np.reshape(model.standardized_forecasts_error, len(df_train_test[nation_list[idx]][0]['GDP']))
        print(f"DW statistic for standardized residuals of {nation_list[idx]}'s arima model: {durbin_watson(stand_resid)}")
        display(acorr_ljungbox(stand_resid, lags = 10))
        print(f"JB p-value for standardized residuals of {nation_list[idx]}'s arima mode: (useless, too few samples) {jarque_bera(stand_resid).pvalue}")
        print('-------------------------------------------------------------------------------')

def arima_prediction_plot(arima_model_list, nation_list, order_list, df_train_test):
    arima_prediction_list = []
    fig, ax = plt.subplots(5, 1, figsize = (15, 15))
    plt.suptitle('Arimax predictions', fontsize = 40)
    plt.tight_layout(pad = 2.5)

    for idx, model in enumerate(arima_model_list):
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(df_train_test[nation_list[idx]][0].drop('GDP', axis=1))
        GDP_test_scaled = scaler.fit_transform(pd.DataFrame(df_train_test[nation_list[idx]][1]['GDP']))
        prediction = model.get_prediction(start = 0, end = 10,
                                          exog = X_test_scaled,
                                          dynamic = False
                                          )
        df_pred = prediction.summary_frame()
        # Reverse scaling for predictions
        df_pred['mean'] = scaler.inverse_transform(df_pred[['mean']])
        arima_prediction_list.append(df_pred['mean'])
        #ax[idx].figure(figsize = (15, 5))
        ax[idx].set_title(f'ARIMA{order_list[idx]} model for {nation_list[idx]} GDP')

        ax[idx].plot(df_train_test[nation_list[idx]][0]['GDP'], '-b', label = 'Data Train')
        #plt.plot(df_train_test['Finland'][0]['GDP'].index, inverse_fitted, 'orange', label = 'In-sample predictions')
        ax[idx].plot(df_train_test[nation_list[idx]][1]['GDP'].index, df_pred['mean'],'-k',label = 'Out-of-sample forecasting')
        ax[idx].plot(df_train_test[nation_list[idx]][1]['GDP'], label = 'Data Test')

        ax[idx].set_xlabel('Time')
        ax[idx].set_ylabel('Values')
        ax[idx].legend(loc = 'upper left')

    plt.show()
    return arima_prediction_list

def add_metrics(model_name:str, model_list, metrics_list, df_train_test, nation_list, prediction_list):
    for idx, model in enumerate(model_list):
        metrics = pd.Series({'Model_name': model_name, 'AIC':model.aic, 
                            'RMSE': root_mean_squared_error(df_train_test[nation_list[idx]][1]['GDP'], prediction_list[idx]),
                            'MAE': mean_absolute_error(df_train_test[nation_list[idx]][1]['GDP'], prediction_list[idx]), 
                            'MAPE':mean_absolute_percentage_error(df_train_test[nation_list[idx]][1]['GDP'], prediction_list[idx])})
        metrics_list[idx] = pd.concat([metrics_list[idx], metrics.to_frame().T])
    return metrics_list