import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.signal import periodogram, find_peaks
import matplotlib.pyplot as plt
from seaborn import lineplot, kdeplot, histplot, boxplot
from statsmodels.api import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import jarque_bera
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.linear_model import LinearRegression

def preprocess_df(df : pd.DataFrame):
    """
    Preprocesses the dataframe removing the nations that don't have all the required columns/values.

    Paramenters:
        df (pandas dataframe): DataFrame with macroeconomic indicators evolution over the years

    Returns:
        tuple: The dataframe with only the valid nations and a list of strings of valid nations
    """
    df = df.replace('China, Hong Kong SAR', 'Hong Kong')
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

def nation_input(number : int, valid_nations : list, df : pd.DataFrame):
    """
    Ask the user for a nation
    """   
    adj_list = ['first', 'second', 'third', 'fourth', 'fifth']
    nat_list = ['Finland', 'Sweden', 'Portugal', 'Poland', 'Germany']
    while True:
        nation = input(f'Please input {adj_list[number]} nation. Press Enter to use the default nation ({nat_list[number]})')
        nation = nation.strip().title()

        if not nation and number == 0:
            nation = 'Finland'
        elif not nation and number == 1:
            nation = 'Sweden'
        elif not nation and number == 2:
            nation = 'Portugal'
        elif not nation and number == 3:
            nation = 'Poland'
        elif not nation and number == 4:
            nation = 'Germany'
        
        if nation in valid_nations:
            return nation
        elif nation in df['Country'].unique() and nation not in valid_nations:
            print(f'Not enough data for {nation}.\nPlease enter a valid nation.\n-----------------')
        else:
            print(f'{nation} is either misspelled or not a nation.\nPlease enter a valid nation.\n-----------------')

def create_nation_list(valid_nations : list, df : pd.DataFrame): 
    """
    Function asking the user for an input about the nation he wants to analyze that must be inside the 
    list of valid nations.

    Parameters:
        valid_nations (list): a list of valid nations that the input must be inside of
        df (pandas DataFrame): the dataframe containing the values of the macroeconomic indicators
    
    Returns:
        list: a list of the selected nations to analyze
    """
    descr = input("The notebook will ask for 5 different nations that you want to perform the analysis about GDP forecasting. A list of valid nations can be found in the file valid_nations.csv in the GitHub page of this project. Any wrong input or nation that isn't included in that file will not be accepted.\n\n Press Enter to continue.")
    nation_list = []
    for nation_num in range(5): 
        nation = nation_input(nation_num, valid_nations, df) 
        nation_list.append(nation)
    return nation_list

def create_df(df : pd.DataFrame):
    """
    Transforms the input dataframe in a dataframe with time as index and 5 macroeconomic indicators as 
    columns. The indicators are: 'Construction (ISIC F)', 'Final consumption expenditure', 'Gross Domestic 
    Product (GDP)', 'General government final consumption expenditure', 'Gross capital formation'

    Parameters:
        df (pandas DataFrame): dataframe to be transformed. It must be of only one nation
    
    Returns:
        DataFrame: transformed dataframe 
    """
    df = df.reset_index().drop(['Country', 'CountryID'], axis = 1)
    df = df.T.reset_index().drop(['index'], axis = 1)
    df.columns = df.loc[0]
    df = df.drop(0)
    df.index = pd.date_range(start="1970-01-01", end="2020-12-31", freq="YS")
    df = df[['Construction (ISIC F)', 'Final consumption expenditure', 'Gross Domestic Product (GDP)', 'General government final consumption expenditure', 'Gross capital formation']]
    df = df.iloc[:, [0, 1, 3, 4, 2]]
    df.columns = ['Construction', 'Consumption', 'Gov_exp', 'Gross_capital', 'GDP']
    return df

def highest_corr_variable(corr_list : list, nation_list : list):
    """
    Displays the column that has highest correlation with the GDP column.

    Parameters:
        corr_list (list): list containing the correlation dataframes of the 5 nations

        nation_list (list): list containing the strings of the 5 nations

    Returns:
        None
    """
    max_list = []
    for corr in corr_list:
        max_corr = corr['GDP'].drop('GDP').max()
        idx = corr.index[corr['GDP'] == max_corr].tolist()[0]
        max_list.append(idx)
    df = pd.DataFrame(max_list, index = nation_list, columns = ['Highest correlation variable'])
    display(df)

def lowest_corr_variable(corr_df : pd.DataFrame):
    """
    Given a dataframe about correlation, it outputs the columns that have the lowest correlation 
    between each other

    Parameters:
        corr_df (pandas DataFrame): dataframe containing the correlations between macroeconomic 
        indicators of a nation

    Returns:
        list: list containing tuples of the nations' lowest correlation variables with each other
    """
    cgdpd = corr_df.drop('GDP', axis = 1).replace({1 : np.nan})
    cgdpd = cgdpd.apply(lambda x : abs(x))
    idx = cgdpd.stack().idxmin()
    return idx

def train_test_split(df : pd.DataFrame):
    """
    Splits the dataframe into a train set from 1970 to 2009 and a test set from 2010 to 2020

    Parameters:
        df (pandas DataFrame): dataframe containing the values of the indicators and time as index

    Returns:
        list: list containing the train dataframe and the test dataframe
    """
    train = df.loc[:'2009-01-01']
    train = train.astype('float')
    test = df.loc['2010-01-01':]
    test = test.astype('float')
    lista = [train, test]
    return lista

def plotseasonal(res, axes):
    """
    Plot the original time series with its trend, seasonality and residuals

    Parameters:
        res (DecomposeResult): the DecomposeResult object resulting from seasonal_decompose

        axes (Axes): Axes matplotlib object in which to plot the decomposition 
    
    Returns:
        None
    """
    res.observed.plot(ax=axes[0], legend=False)
    axes[0].set_ylabel('Observed')
    res.trend.plot(ax=axes[1], legend=False)
    axes[1].set_ylabel('Trend')
    res.seasonal.plot(ax=axes[2], legend=False)
    axes[2].set_ylabel('Seasonal')
    res.resid.plot(ax=axes[3], legend=False)
    axes[3].set_ylabel('Residual')

def roll_mean_std_plot(df : dict, nation_list : list, roll_num : int, cmap : list):
    """
    Plot a time series with its rolling mean and standard deviation

    Paramenters:
        df (dictionary): dictionary with a nation as key, and as value a list of 2 pandas dataframes 
        (train and test)

        nation_list (list): list containing the strings of the 5 nations

        roll_num (int): how many points to consider while calculating the rolling mean and std

        cmap (list): list of html color codes

    Returns:
        None
    """
    fig, ax = plt.subplots(len(nation_list), 1, figsize = (15, 15))
    plt.suptitle('GDP of the nations', fontsize = 40)
    plt.tight_layout(pad = 2.5)

    for idx, nation in enumerate(nation_list):
        rolling_mean = df[nation][0]['GDP'].rolling(roll_num).mean()
        rolling_std = df[nation][0]['GDP'].rolling(roll_num).std()
        lineplot(df[nation][0]['GDP'], ax = ax[idx], c = cmap[3])
        ax[idx].plot(rolling_mean, color = cmap[0], label = 'Rolling Mean')
        ax[idx].plot(rolling_std, color = cmap[1], label = 'Rolling Std')
        ax[idx].set_title(nation, fontsize = 20)
        ax[idx].legend(loc = 'best')

    plt.show()

def check_stationarity(df_train_test : dict, nation_list : list, variable : str):
    """
    Checks whether the train sets of the provided time series are stationary, according to the ADFuller 
    test and the KPSS test

    Parameters:
        df_train_test (dictionary): dictionary with a nation as key, and as value a list of 2 pandas 
        dataframes (train and test)

        nation_list (list): list containing the strings of the 5 nations

    Returns:
        DataFrame: A dataframe displaying statistics about stationarity tests of the various nations
    """
    list_for_df = []
    for idx, nation in enumerate(nation_list):
        adf_test = adfuller(df_train_test[nation][0][variable], autolag='AIC')
        lista_prova = []
        lista_prova.append(adf_test[0])
        lista_prova.append(adf_test[1])
        if adf_test[1] <= 0.05:
            lista_prova.append('Yes')
        else:
            lista_prova.append('No')
        kpss_out = kpss(df_train_test[nation][0][variable], regression='c', nlags='auto', store=True)
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

def difference(df_train_test : dict, difference : int, nation : str):
    """
    Performs differencing of a time series

    Parameters:
        df_train_test (dictionary): dictionary with a nation as key, and as value a list of 2 pandas 
        dataframes (train and test)

        difference (int): number of the lag to perform the differencing with. A difference equal to 1 
        performs first order differencing

        nation (str): the nation to perform the differencing with
    
    Returns:
        tuple: tuple containing the differenced train set and the differenced test set
    """
    full_df = pd.concat([df_train_test[nation][0], df_train_test[nation][1]])
    diff_df = full_df.diff(difference).dropna()
    tuple = train_test_split(diff_df)
    return tuple

def detrend(df_train_test_log_dif : dict, nation_list : list, seasons_list : list):
    df_train_test_log_dif_detrend = {}
    for idx, nation in enumerate(nation_list):
        stl = STL(df_train_test_log_dif[nation][0]['GDP'], period = seasons_list[idx])
        result = stl.fit()
        trend = result.trend
        df_train_test_log_dif_detrend[nation] = df_train_test_log_dif[nation][0]['GDP'] - trend
    return df_train_test_log_dif_detrend

def psd(nation : str, df_train_test : dict, Fs : int):
    """
    Plot the periodogram of a time series with its prominent peaks and display a dataframe with the 
    most important periods

    Paramenters:
        nation (str): nation to perform the power spectral analysis with

        df_train_test (dictionary): dictionary with a nation as key, and as value a list of 2 pandas 
        dataframes (train and test)

        Fs (int): sampling rate

    Returns:
        int: best seasonality for the time series
    """
    f_per, Pxx_per = periodogram(df_train_test[nation][0]['GDP'], Fs, detrend = None, window = 'triang', return_onesided = True, scaling = 'density')
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

# MODELS

def create_metrics_df():
    """
    Creates an empty dataframe for the 5 nations where the metrics of the models will be stored. 

    Returns:
        list: list of empty dataframes to store the metrics
    """
    df_metrics_1 = pd.DataFrame(columns = ['Model_name','AIC','MAE','RMSE','MAPE'])
    df_metrics_2 = pd.DataFrame(columns = ['Model_name','AIC','MAE','RMSE','MAPE'])
    df_metrics_3 = pd.DataFrame(columns = ['Model_name','AIC','MAE','RMSE','MAPE'])
    df_metrics_4 = pd.DataFrame(columns = ['Model_name','AIC','MAE','RMSE','MAPE'])
    df_metrics_5 = pd.DataFrame(columns = ['Model_name','AIC','MAE','RMSE','MAPE'])
    metrics_list_of_df = [df_metrics_1, df_metrics_2, df_metrics_3, df_metrics_4, df_metrics_5]
    return metrics_list_of_df

def slr_prediction_plot(nation_list : list, df_train_test : dict):
    """
    Fits a simple linear regression for every nation in the provided list, then displays a plot with 
    the train set, test set, fitted values and prediction of the model.

    Parameters:
        nation_list (list): list containing the strings of the 5 nations

        df_train_test (dictionary): dictionary with a nation as key, and as value a list of 2 pandas 
        dataframes (train and test)

    Returns:
        list: list of the fitted models

        list: list of the predictions done by the models on the test set

        list: list of the AICs of the models
    """
    slr_model_list = []
    slr_prediction_list = []
    aic_list = []
    fig, ax = plt.subplots(5, 1, figsize = (15, 18))
    plt.suptitle('Simple linear regression predictions', fontsize = 40)
    plt.tight_layout(pad = 2.5)

    for idx, nation in enumerate(nation_list):
        model = LinearRegression()
        X_train = pd.DataFrame(df_train_test[nation][0]['GDP'].index.year)
        y_train = df_train_test[nation][0]['GDP']
        X_test = pd.DataFrame(df_train_test[nation][1]['GDP'].index.year)
        y_test = df_train_test[nation][1]['GDP']

        model.fit(X_train, y_train)
        y_fit = model.predict(X_train)
        y_fit = pd.Series(y_fit, index = y_train.index)
        y_pred = model.predict(X_test)
        y_pred = pd.Series(y_pred, index = y_test.index)

        rss = np.sum((y_train - y_fit) ** 2)
        n = X_train.shape[0]
        k = X_train.shape[1] + 1 
        sigma_squared = mean_squared_error(y_train, y_fit)
        log_likelihood = -n / 2 * (np.log(2 * np.pi * sigma_squared) + 1)
        aic = 2 * k - 2 * log_likelihood

        slr_model_list.append(model)
        slr_prediction_list.append(y_pred)
        aic_list.append(aic)

        ax[idx].set_title(f"Linear regression model for {nation}'s GDP")

        ax[idx].plot(y_pred, '-k', label = 'Out-of-sample forecasting')
        ax[idx].plot(y_test, label = 'Data Test')
        ax[idx].plot(y_train, '-b', label = 'Data Train')
        ax[idx].plot(y_fit, 'orange', label = 'In-sample predictions')

        ax[idx].set_xlabel('Time')
        ax[idx].set_ylabel('Values')
        ax[idx].legend(loc = 'upper left')

    plt.show()
    return slr_model_list, slr_prediction_list, aic_list

def res_diagnostics_plot(residuals : np.ndarray, nation : str):
    """
    Plots some important statistics for the residuals of a model.

    Parameters:
        residuals (array): array with the residuals of a model

        nation (str): nation of the model to perform the diagnostics

    Returns:
        None
    """
    #residuals = (residuals - np.mean(residuals)) / np.std(residuals)
    fig, ax = plt.subplots(2, 2, figsize = (12, 8))
    fig.tight_layout(pad = 5, h_pad = 4)
    fig.suptitle(f'Residual diagnostic plot for {nation}', fontsize = 20)

    norm = np.random.normal(size = 100000)
    kdeplot(ax = ax[0][0], x = norm, color = '#1d4863', label = 'N(0,1)')
    kdeplot(ax = ax[0][0], x = residuals, color = 'teal', label = 'Hist')
    histplot(ax = ax[0][0], x = residuals, stat = 'density', linewidth = 0.5, kde = True, color = 'teal', bins = 15)
    ax[0][0].set(title = 'Histogram of residuals', xlabel = 'Residual')
    ax[0][0].legend()

    boxplot(ax = ax[0][1], x = residuals, showmeans = True);
    ax[0][1].set(title = 'Boxplot of residuals', xlabel = 'Residual')

    qqplot(residuals, line = 'q', ax = ax[1][0]);
    ax[1][0].set_title('Q-Q plot')

    plot_acf(residuals, ax = ax[1][1], lags = 10)
    ax[1][1].set(xlabel = 'Lags')
    ax[1][1].set_title('Correlogram')

    plt.show()

def mlr_prediction_plot(nation_list : list, df_train_test : dict):
    """
    Fits a multiple linear regression for every nation in the provided list, then displays a plot with 
    the train set, test set, fitted values and prediction of the model.

    Parameters:
        nation_list (list): list containing the strings of the 5 nations

        df_train_test (dictionary): dictionary with a nation as key, and as value a list of 2 pandas 
        dataframes (train and test)

    Returns:
        list: list of the fitted models

        list: list of the predictions done by the models on the test set

        list: list of the AICs of the models
    """
    mlr_model_list = []
    mlr_prediction_list = []
    aic_list = []
    fig, ax = plt.subplots(5, 1, figsize = (15, 18))
    plt.suptitle('Linear regression predictions', fontsize = 40)
    plt.tight_layout(pad = 2.5)

    for idx, nation in enumerate(nation_list):
        model = LinearRegression()
        X_train = df_train_test[nation][0].drop('GDP', axis = 1)
        y_train = df_train_test[nation][0]['GDP']
        X_test = df_train_test[nation][1].drop('GDP', axis = 1)
        y_test = df_train_test[nation][1]['GDP']

        model.fit(X_train, y_train)
        y_fit = model.predict(X_train)
        y_fit = pd.Series(y_fit, index = y_train.index)
        y_pred = model.predict(X_test)
        y_pred = pd.Series(y_pred, index = y_test.index)

        rss = np.sum((y_train - y_fit) ** 2)
        n = X_train.shape[0]
        k = X_train.shape[1] + 1 
        sigma_squared = mean_squared_error(y_train, y_fit)
        log_likelihood = -n / 2 * (np.log(2 * np.pi * sigma_squared) + 1)
        aic = 2 * k - 2 * log_likelihood

        mlr_model_list.append(model)
        mlr_prediction_list.append(y_pred)
        aic_list.append(aic)

        ax[idx].set_title(f'Linear regression {nation} model for {nation} GDP')

        ax[idx].plot(y_pred, '-k', label = 'Out-of-sample forecasting')
        ax[idx].plot(y_test, label = 'Data Test')
        ax[idx].plot(y_train, '-b', label = 'Data Train')
        ax[idx].plot(y_fit, 'orange', label = 'In-sample predictions')

        ax[idx].set_xlabel('Time')
        ax[idx].set_ylabel('Values')
        ax[idx].legend(loc = 'upper left')

    plt.show()
    return mlr_model_list, mlr_prediction_list, aic_list

def arima_order(nation_list : list, df_train_test : dict):
    """
    Finds the best "p", "d" and "q" parameters for an ARIMAX model for every nation in the provided 
    list according to the AIC statistic, and then fits a model with these.

    Parameters:
        nation_list (list): list containing the strings of the 5 nations

        df_train_test (dictionary): dictionary with a nation as key, and as value a list of 2 pandas 
        dataframes (train and test)

    Returns:
        list: list containing the orders of the various models

        list: list containing the fitted models     
    """
    order_list = []
    arima_model_list = []
    for idx, nation in enumerate(nation_list):
        print(nation)

        best_model = auto_arima(
        df_train_test[nation][0]['GDP'], 
        X = df_train_test[nation][0].drop('GDP', axis = 1),
        start_p = 0, d = 1, start_q = 0, 
        max_p = 4, max_q = 4,
        seasonal = False,
        error_action = 'warn', 
        with_intercept = True, 
        trace = True, 
        suppress_warnings = True,
        stepwise = True,
        random_state = 20, 
        information_criterion = 'aic',
        disp = False
        )

        order_list.append(best_model.order)

        arimax_model = ARIMA(df_train_test[nation][0]['GDP'], 
                             exog = df_train_test[nation][0].drop('GDP', axis = 1),
                             order = best_model.order
                             ).fit()
        
        arima_model_list.append(arimax_model)
    return order_list, arima_model_list

def res_stats(model_list : list, nation_list : list, df_train_test : dict, lr = False, residuals : np.ndarray = None):
    """
    For every nation in the provided list, displays some statistics about the residuals of the given 
    model.

    Parameters:
        model_list: list of the models to calculate the statistics from

        nation_list (list): list containing the strings of the 5 nations

        df_train_test (dictionary): dictionary with a nation as key, and as value a list of 2 pandas 
        dataframes (train and test)

    Returns:
        None
    """
    if lr == True:
        for idx, model in enumerate(model_list):
            stand_resid = np.reshape(residuals, len(df_train_test[nation_list[idx]][0]['GDP']))
            print(f"DW statistic for standardized residuals of {nation_list[idx]}'s model: {durbin_watson(stand_resid)}")
            display(acorr_ljungbox(stand_resid, lags = 10))
            print(f"JB p-value for standardized residuals of {nation_list[idx]}'s model: (useless, too few samples) {jarque_bera(stand_resid).pvalue}")
            print('-------------------------------------------------------------------------------\n')
    else:
        for idx, model in enumerate(model_list):
            stand_resid = np.reshape(model.standardized_forecasts_error, len(df_train_test[nation_list[idx]][0]['GDP']))
            print(f"DW statistic for standardized residuals of {nation_list[idx]}'s model: {durbin_watson(stand_resid)}")
            display(acorr_ljungbox(stand_resid, lags = 10))
            print(f"JB p-value for standardized residuals of {nation_list[idx]}'s model: (useless, too few samples) {jarque_bera(stand_resid).pvalue}")
            print('-------------------------------------------------------------------------------\n')

def arima_prediction_plot(arima_model_list : list, nation_list : list, order_list : list, df_train_test : dict):
    """
    For every nation plots its train set, test set, fitted values and predictions of an ARIMAX model.

    Parameters:
        arima_model_list: list of the ARIMAX models

        nation_list (list): list containing the strings of the 5 nations

        order_list (list): list with the orders of the models

        df_train_test (dictionary): dictionary with a nation as key, and as value a list of 2 pandas 
        dataframes (train and test)
   
    """
    arima_prediction_list = []
    fig, ax = plt.subplots(5, 1, figsize = (15, 18))
    plt.suptitle('Arimax predictions', fontsize = 40)
    plt.tight_layout(pad = 2.5)

    for idx, model in enumerate(arima_model_list):
        prediction = model.get_prediction(start = '2010', end = '2020',
                                          exog = df_train_test[nation_list[idx]][1].drop('GDP', axis = 1),
                                          dynamic = False
                                          )
        df_pred = prediction.summary_frame()
        pred_ci_original = prediction.conf_int()
        arima_prediction_list.append(df_pred['mean'])
        ax[idx].set_title(f'ARIMA{order_list[idx]} model for {nation_list[idx]} GDP')

        ax[idx].plot(df_train_test[nation_list[idx]][0]['GDP'], '-b', label = 'Data Train')
        ax[idx].plot(model.fittedvalues, 'orange', label = 'In-sample predictions')
        ax[idx].plot(df_pred['mean'],'-k', label = 'Out-of-sample forecasting')
        ax[idx].plot(df_train_test[nation_list[idx]][1]['GDP'], label = 'Data Test')
        ax[idx].fill_between(
            df_train_test[nation_list[idx]][1]['GDP'].index, 
            pred_ci_original['lower GDP'], 
            pred_ci_original['upper GDP'], 
            color = 'blue', alpha = 0.2, label='Confidence Interval'
        )

        ax[idx].set_xlabel('Time')
        ax[idx].set_ylabel('Values')
        ax[idx].legend(loc = 'upper left')

    plt.show()
    return arima_prediction_list

def add_metrics(model_name : str, model_list : list, metrics_df : pd.DataFrame, df_train_test : dict, nation_list : list, prediction_list : list, lr = False, aic_list : list = None):
    """
    Add the metrics Akaike information criterion, root mean squared error, mean absolute error, mean 
    absolute percentate error to a pre-existing dataframe.

    Parameters:
        model_name (str): the name of the model to display in the dataframe

        model_list (list): list of the models to calculate the metrics

        metrics_df (dataframe): existing dataframe with metrics of previous models

        df_train_test (dictionary): dictionary with a nation as key, and as value a list of 2 pandas 
        dataframes (train and test)

        nation_list (list): list containing the strings of the 5 nations

        ls (bool, default False): set to True if the provided models are simple or multiple linear 
        regressions

        aic_list (list, default None): a list of the AIC criterions of the models. Need to be provided 
        if the models are simple or multiple linear regressions

    Returns:
        dataframe: updated pandas dataframe with the added metrics
    """
    if lr:
        for idx, model in enumerate(model_list):
            mae = round(mean_absolute_percentage_error(df_train_test[nation_list[idx]][1]['GDP'], prediction_list[idx]) * 100, 2)
            metrics = pd.Series({'Model_name': model_name, 'AIC': round(aic_list[idx]), 
                                'RMSE': round(root_mean_squared_error(df_train_test[nation_list[idx]][1]['GDP'], prediction_list[idx])),
                                'MAE': round(mean_absolute_error(df_train_test[nation_list[idx]][1]['GDP'], prediction_list[idx])), 
                                'MAPE': mae})
            metrics_df[idx] = pd.concat([metrics_df[idx], metrics.to_frame().T])
        return metrics_df
    else:
        for idx, model in enumerate(model_list):
            mae = round(mean_absolute_percentage_error(df_train_test[nation_list[idx]][1]['GDP'], prediction_list[idx]) * 100, 2)
            metrics = pd.Series({'Model_name': model_name, 'AIC': round(model.aic), 
                                'RMSE': round(root_mean_squared_error(df_train_test[nation_list[idx]][1]['GDP'], prediction_list[idx])),
                                'MAE': round(mean_absolute_error(df_train_test[nation_list[idx]][1]['GDP'], prediction_list[idx])), 
                                'MAPE': mae})
            metrics_df[idx] = pd.concat([metrics_df[idx], metrics.to_frame().T])
        return metrics_df

def ets_order(df_train_test : dict, nation_list : list):
    """
    Performs a grid search to the train time series of the dataframe to find the optimal "error", 
    "trend", "seasonal", "seasonal_periods", "damped_tred" parameters of an ETS model.

    Parameters:
        df_train_test (dictionary): dictionary with a nation as key, and as value a list of 2 pandas 
        dataframes (train and test)

        nation_list (list): list containing the strings of the 5 nations
    
    Returns:
        list: list of fitted ETS models for the nations

        dataframe: pandas dataframe with the parameters of the ETS models
    """
    df_params = []
    model_list = []
    for nation in nation_list:
        best_map = 100
        best_i = 0
        for i in range(2, 16):
            for err in ['add', 'mul']:
                for trend in ['add', 'mul']:
                    for season in ['add', 'mul']:
                        for dam_trend in [True, False]:
                            model_1 = ETSModel(df_train_test[nation][0]['GDP'],
                                            error = err,
                                            trend = trend, 
                                            seasonal = season, 
                                            seasonal_periods = i, 
                                            damped_trend = dam_trend).fit(disp = False)
                            pred_1 = model_1.get_prediction(start = '2010', end = '2020')
                            df_1 = pred_1.summary_frame()
                            if df_1['mean'].isnull().sum() == 0:
                                map = mean_absolute_percentage_error(df_train_test[nation][1]['GDP'], df_1['mean'])
                            else:
                                pass
                        if(map < best_map):
                            best_map = map
                            best_err = err
                            best_trend = trend
                            best_season = season
                            best_i = i
                            best_dam_trend = dam_trend
                            best_model = ETSModel(df_train_test[nation][0]['GDP'], 
                                                  error = best_err,
                                                  trend = best_trend, 
                                                  seasonal = best_season, 
                                                  seasonal_periods = best_i,
                                                  damped_trend = best_dam_trend).fit(disp = False)
        model_list.append(best_model)
        x = [best_err, best_trend, best_season, best_i, best_dam_trend]
        df_params.append(x)
    df_params = pd.DataFrame(df_params, index = nation_list, columns = ['Error', 'Trend', 'Seasonal', 'Seasonal period', 'Damped trend'])

    return model_list, df_params

def ets_prediction_plot(ets_model_list : list, nation_list : list, df_train_test : dict):
    """
    For every nation plots its train set, test set, fitted values and predictions of an ETS model.

    Parameters:
        ets_model_list (list): list of the ETS models

        nation_list (list): list containing the strings of the 5 nations

        df_train_test (dictionary): dictionary with a nation as key, and as value a list of 2 pandas 
        dataframes (train and test)

    Returns:
        list: list of the predictions made by the fitted ETS models
    """
    ets_prediction_list = []
    fig, ax = plt.subplots(5, 1, figsize = (15, 18))
    plt.suptitle('Ets predictions', fontsize = 40)
    plt.tight_layout(pad = 2.5)

    for idx, model in enumerate(ets_model_list):
        prediction = model.get_prediction(start = '2010', end = '2020')
        df_pred = prediction.summary_frame()
        ets_prediction_list.append(df_pred['mean'])
        ax[idx].set_title(f'Ets model for {nation_list[idx]} GDP')

        ax[idx].plot(df_train_test[nation_list[idx]][0]['GDP'], '-b', label = 'Data Train')
        ax[idx].plot(model.fittedvalues, 'orange', label = 'In-sample predictions')
        ax[idx].plot(df_train_test[nation_list[idx]][1]['GDP'].index, df_pred['mean'],'-k',label = 'Out-of-sample forecasting')
        ax[idx].plot(df_train_test[nation_list[idx]][1]['GDP'], label = 'Data Test')
        ax[idx].fill_between(
            df_train_test[nation_list[idx]][1]['GDP'].index, 
            df_pred['pi_lower'], 
            df_pred['pi_upper'], 
            color = 'blue', alpha = 0.2, label='Confidence Interval'
        )

        ax[idx].set_xlabel('Time')
        ax[idx].set_ylabel('Values')
        ax[idx].legend(loc = 'upper left')

    plt.show()
    return ets_prediction_list

def grangers_causation(data, name_variables, test, maxlags):
    """
    
    """
    df = pd.DataFrame(np.zeros((len(name_variables), len(name_variables))), columns = name_variables, index = name_variables)
    for c in df.columns:
        for r in df.columns:
            test_result = grangercausalitytests(data[[r, c]], [maxlags], verbose = False)
            p_values = round(test_result[maxlags][0][test][1], 4)
            df.loc[c,r] = p_values
    return df

def grangers_causation_columns(df_train_test_log_dif : dict, nation_list : list):
    """
    For each nation in the nation_list, perform Granger causality tests and return columns that
    Granger cause each other while ensuring GDP is part of the relationship.

    Parameters:
        df_train_test_log_dif (dict): Dictionary of time series data for each nation.

        nation_list (list): List of nation names.

    Returns:
        list: list of columns that granger cause each other
    """
    grangers_columns = []
    for nation in nation_list:
        print(f"Processing {nation}")
        
        model = VAR(df_train_test_log_dif[nation][0])
        lag_selection = model.select_order(maxlags=5, trend='ctt')
        optimal_lag = lag_selection.aic

        df = grangers_causation(df_train_test_log_dif[nation][0], df_train_test_log_dif[nation][0].columns, 'ssr_chi2test', optimal_lag)
        
        causally_related_columns = set()
        for col in df.columns:
            for row in df.index:
                if df.loc[row, col] < 0.05 and 'GDP' in [row, col]:
                    causally_related_columns.add(row)
                    causally_related_columns.add(col)
        
        grangers_columns.append(list(causally_related_columns))
        print(f"Columns causally related to GDP and each other: {list(causally_related_columns)}")
    
    return grangers_columns


def varma1_order(df_train_test_log_dif : dict, nation_list : list, grangers_causation_columns: list):
    """
    Finds the best VARX(p) for each nation using AIC. Only use a selected number of columns 
    as endogenous variables, and the remaining as exogenous. If the number of endogenous variables is 
    the same as the total number of variables in the dataframe, only use endogenous variable and no 
    hexogenous, effectively fitting a VAR model.
    
    Parameters:
        df_train_test_log_dif (dict): Dictionary of time series data for each nation.

        nation_list (list): List of nation names.

        grangers_causation_columns (list): List of lists containing the selected columns for each nation.
    
    Returns:
        tuple: A list of (p) parameters and a list of fitted VARX models for each nation.
    """
    varmax_model_list = []
    results = []

    for idx, nation in enumerate(nation_list):
        print(f"Processing nation: {nation}")
        best_aic = float("inf")
        best_p, best_q = None, None
        best_model = None

        if not grangers_causation_columns[idx]:
            print(f"No causal columns for {nation}, skipping.")
            results.append(None)
            varmax_model_list.append(None)
            continue

        if len(grangers_causation_columns) == len(df_train_test_log_dif[nation][0].columns):
            for p in range(1, 6):
                    try:
                        model = VARMAX(
                            df_train_test_log_dif[nation][0][grangers_causation_columns[idx]],
                            #exog = df_train_test_log_dif[nation][0].drop(grangers_causation_columns[idx], axis = 1),
                            order=(p, 0)
                        ).fit(disp=False)
                        aic = model.aic
                        if aic < best_aic:
                            best_aic = aic
                            best_p = p
                            best_model = model
                    except Exception as e:
                        print(f"Error fitting VARMAX for {nation}, (p)=({p}): {e}")
                        continue
        if len(grangers_causation_columns) != len(df_train_test_log_dif[nation][0].columns):      
            for p in range(1, 6):
                    try:
                        model = VARMAX(
                            df_train_test_log_dif[nation][0][grangers_causation_columns[idx]],
                            exog = df_train_test_log_dif[nation][0].drop(grangers_causation_columns[idx], axis = 1),
                            order=(p, 0)
                        ).fit(disp=False)
                        aic = model.aic
                        if aic < best_aic:
                            best_aic = aic
                            best_p = p
                            best_model = model
                    except Exception as e:
                        print(f"Error fitting VARMAX for {nation}, (p)=({p}): {e}")
                        continue
        
        if best_model is not None:
            varmax_model_list.append(best_model)
            results.append([best_p])
            print(f"Best order for {nation}: (p)=({best_p}), AIC={best_aic}")
        else:
            print(f"No valid model found for {nation}.")
            varmax_model_list.append(None)
            results.append(None)
    
    return results, varmax_model_list

def var_res_stats(model_list : list, nation_list : list, df_train_test : dict):
    """
    For every nation in the provided list, displays some statistics about the residuals of the given model. 

    Parameters:
        model_list: list of the models to calculate the statistics from

        nation_list (list): list containing the strings of the 5 nations

        df_train_test (dictionary): dictionary with a nation as key, and as value a list of 2 pandas dataframes (train and test)

    Returns:
        None
    """
    for idx, model in enumerate(model_list):
        stand_resid = np.reshape(model.standardized_forecasts_error[-1], len(df_train_test[nation_list[idx]][0]['GDP']))
        print(f"DW statistic for standardized residuals of {nation_list[idx]}'s model: {durbin_watson(stand_resid)}")
        display(acorr_ljungbox(stand_resid, lags = 10))
        print(f"JB p-value for standardized residuals of {nation_list[idx]}'s model: (useless, too few samples) {jarque_bera(stand_resid).pvalue}")
        print('-------------------------------------------------------------------------------')

def invert_first_order_differencing(differenced_series : pd.Series, original_series : pd.Series):
    """
    Inverts a first-order differencing to reconstruct the original series.
    
    Parameters:
        differenced_series (pd.Series): The differenced series (e.g., output of `series.diff()`).

        original_series (pd.Series): The original series (must include the first value of the original series).
    
    Returns:
        pd.Series: The reconstructed original series.
    """
    if not isinstance(differenced_series, pd.Series) or not isinstance(original_series, pd.Series):
        raise ValueError("Both inputs must be pandas Series.")
    
    inverted_series = differenced_series.cumsum() + original_series.iloc[0]
    return inverted_series

def varma1_prediction_plot(varma_model_list : list, nation_list : list, order_list : list, df_train_test : dict, df_train_test_log_dif : dict, grangers_col_list : list):
    """
    For every nation plots its train set, test set, fitted values and predictions of a VARX model.

    Parameters:
        varma_model_list (list): list of the VARX models

        nation_list (list): list containing the strings of the 5 nations

        order_list (list): list of the orders of the parameters of the various VARX models

        df_train_test (dictionary): dictionary with a nation as key, and as value a list of 2 pandas dataframes (train and test)

        df_train_test_log_dif (dictionary): dictionary with a nation as key, and as value a list of 2 pandas dataframes (train and test). The time series of these dataframe need to be stationary and of first order differencing of the df_train_test dataframe

        grangers_col_list (list): list of the columns to use as endog variables (GDP excluded)

    Returns:
        list: list of the predictions made by the fitted VARX models
    """
    varma_prediction_list = []
    fig, ax = plt.subplots(5, 1, figsize = (15, 18))
    plt.suptitle('Varma predictions', fontsize = 40)
    plt.tight_layout(pad = 2.5)

    for idx, model in enumerate(varma_model_list):
        prediction = model.get_prediction(start = '2010', end = '2020',
                                          exog = df_train_test_log_dif[nation_list[idx]][1].drop(grangers_col_list[idx], axis = 1),
                                          dynamic = False)
        df_pred = prediction.summary_frame()
        pred_ci = prediction.conf_int()
        df_pred['mean'] = invert_first_order_differencing(df_pred['mean'], df_train_test[nation_list[idx]][1]['GDP'])
        inverse_fitted = invert_first_order_differencing(model.fittedvalues['GDP'], df_train_test[nation_list[idx]][0]['GDP'])
        pred_ci_original = pd.DataFrame({
            'lower': invert_first_order_differencing(pred_ci.loc[:, 'lower GDP'], df_train_test[nation_list[idx]][1]['GDP']),
            'upper': invert_first_order_differencing(pred_ci.loc[:, 'upper GDP'], df_train_test[nation_list[idx]][1]['GDP'])
})      
        varma_prediction_list.append(df_pred['mean'])

        ax[idx].set_title(f'Varma{order_list[idx]} model for {nation_list[idx]} GDP')

        ax[idx].plot(df_train_test[nation_list[idx]][0]['GDP'], '-b', label = 'Data Train')
        ax[idx].plot(inverse_fitted, 'orange', label = 'In-sample predictions')
        ax[idx].plot(df_pred['mean'],'-k',label = 'Out-of-sample forecasting')
        ax[idx].plot(df_train_test[nation_list[idx]][1]['GDP'], label = 'Data Test')
        ax[idx].fill_between(
            df_train_test['Finland'][1]['GDP'].index, 
            pred_ci_original['lower'], 
            pred_ci_original['upper'], 
            color = 'blue', alpha = 0.2, label = 'Confidence Interval'
        )

        ax[idx].set_xlabel('Time')
        ax[idx].set_ylabel('Values')
        ax[idx].legend(loc = 'upper left')

    plt.show()
    return varma_prediction_list

def varma2_order(df_train_test_log_dif : dict, nation_list : list, grangers_col_list : list):
    """
    Finds the best VAR parameter (p) for each nation using AIC.
    
    Parameters:
        df_train_test_log_dif (dict): Dictionary of time series data for each nation.

        nation_list (list): List of nation names.

        grangers_causation_columns (list): List of lists containing the selected columns for each nation.
    
    Returns:
        tuple: A list of (p) parameters and a list of fitted VARMAX models for each nation.
    """
    varmax_model_list = []
    results = []
    
    for idx, nation in enumerate(nation_list):
        print(f'Processing nation: {nation}')
        best_aic = float("inf")
        best_p, best_q = None, None
        best_model = None
        
        grangers_col_list[idx].remove('GDP')
        for variable in grangers_col_list[idx]:
            for p in range(1, 6):
                try:
                    model = VARMAX(
                        df_train_test_log_dif[nation][0][['GDP', variable]],
                        order = (p, 0)
                    ).fit(disp=False)
                    aic = model.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_p = p
                        best_model = model
                        best_variable = variable
                except Exception as e:
                    print(f"Error fitting VARMAX for {nation}, (p)=({p}) and variable {variable}: {e}")
                    continue
        
        if best_model is not None:
            varmax_model_list.append(best_model)
            results.append([best_p, best_q])
            print(f"Best order for {nation}: (p) = ({best_p}), AIC = {best_aic} variable ({variable})")
        else:
            print(f"No valid model found for {nation}.")
            varmax_model_list.append(None)
            results.append(None)
    
    return results, varmax_model_list

def varma2_prediction_plot(varma_model_list: list, nation_list : list, order_list : list, df_train_test : dict, df_train_test_log_dif : dict):
    """
    For every nation plots its train set, test set, fitted values and predictions of a VARMA model.

    Parameters:
        varma_model_list (list): list of the VARMA models

        nation_list (list): list containing the strings of the 5 nations

        order_list (list): list of the orders of the parameters of the various VARMA models

        df_train_test (dictionary): dictionary with a nation as key, and as value a list of 2 pandas dataframes (train and test)

        df_train_test_log_dif (dictionary): dictionary with a nation as key, and as value a list of 2 pandas dataframes (train and test). The time series of these dataframe need to be stationary and of first order differencing of the df_train_test dataframe

    Returns:
        list: list of the predictions made by the fitted VARMA models
    """
    varma_prediction_list = []
    fig, ax = plt.subplots(5, 1, figsize = (15, 18))
    plt.suptitle('Varma predictions', fontsize = 40)
    plt.tight_layout(pad = 2.5)

    for idx, model in enumerate(varma_model_list):
        prediction = model.get_prediction(start = '2010', end = '2020',
                                          exog = df_train_test_log_dif[nation_list[idx]][1].drop('GDP', axis = 1),
                                          dynamic = False)
        df_pred = prediction.summary_frame()
        pred_ci = prediction.conf_int()
        inverse_fitted = invert_first_order_differencing(model.fittedvalues['GDP'], df_train_test[nation_list[idx]][0]['GDP'])
        df_pred['mean'] = invert_first_order_differencing(df_pred['mean'], df_train_test[nation_list[idx]][1]['GDP'])
        pred_ci_original = pd.DataFrame({
            'lower': invert_first_order_differencing(pred_ci.loc[:, 'lower GDP'], df_train_test[nation_list[idx]][1]['GDP']),
            'upper': invert_first_order_differencing(pred_ci.loc[:, 'upper GDP'], df_train_test[nation_list[idx]][1]['GDP'])
})
        varma_prediction_list.append(df_pred['mean'])

        ax[idx].set_title(f'Varma{order_list[idx]} model for {nation_list[idx]} GDP')

        ax[idx].plot(df_train_test[nation_list[idx]][0]['GDP'], '-b', label = 'Data Train')
        ax[idx].plot(inverse_fitted, 'orange', label = 'In-sample predictions')
        ax[idx].plot(df_pred['mean'],'-k',label = 'Out-of-sample forecasting')
        ax[idx].plot(df_train_test[nation_list[idx]][1]['GDP'], label = 'Data Test')
        ax[idx].fill_between(
            df_train_test['Finland'][1]['GDP'].index, 
            pred_ci_original['lower'], 
            pred_ci_original['upper'], 
            color = 'blue', alpha = 0.2, label = 'Confidence Interval'
        )

        ax[idx].set_xlabel('Time')
        ax[idx].set_ylabel('Values')
        ax[idx].legend(loc = 'upper left')

    plt.show()
    return varma_prediction_list

def best_model(metrics_df : pd.DataFrame, nation_list : list):
    """
    Based on a dataframe with metrics of various models, returns the best model for every criterion. The metrics need to be "AIC", "RMSE", "MAE", "MAPE".

    Parameters:
        metrics_df (pd.DataFrame): data about the metrics of the models

        nation_list (list): list of strings of the nations the model where fitted with

    Returns:
        None
    """
    for idx, df in enumerate(metrics_df):
        df.reset_index(drop = True, inplace = True)
        df = df.drop(df[df['Model_name'] == 'ETS'].index, axis = 0)
        mask = df['AIC'] == df['AIC'].min()
        best_aic = df[mask]['Model_name'].iloc[0]
        mask = df['MAE'] == df['MAE'].min()
        best_mae = df[mask]['Model_name'].iloc[0]
        mask = df['RMSE'] == df['RMSE'].min()
        best_rmse = df[mask]['Model_name'].iloc[0]
        mask = df['MAPE'] == df['MAPE'].min()
        best_mape = df[mask]['Model_name'].iloc[0]
        best_dic = {'Best AIC' : best_aic,
                    'Best MAE' : best_mae,
                    'Best RMSE' : best_rmse,
                    'Best MAPE' : best_mape}
        best_df = pd.DataFrame(best_dic, index = [nation_list[idx]])
        display(best_df)