import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.signal import periodogram, find_peaks
import matplotlib.pyplot as plt
from seaborn import lineplot

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
            first = 'Canada'
        if not first and number == 5:
            first = 'South Africa'
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

def check_stationarity(df_list, df_train_test, nation_list):
    list_for_df = []
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
    return final_df

def log_transform(df_train_test):
    for nation in df_train_test:
        df_train_test[nation] = (np.log(df_train_test[nation][0]), np.log(df_train_test[nation][1]))
    return df_train_test

def difference(df_train_test, difference, nation):
    tuple = (df_train_test[nation][0].diff(difference).dropna(), df_train_test[nation][1].diff().dropna())
    return tuple

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