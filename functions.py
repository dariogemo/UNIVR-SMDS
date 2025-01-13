import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

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

def highest_corr_variable(corr_list, max_list):
    for corr in corr_list:
        max_corr = corr['GDP'].drop('GDP').max()
        idx = corr.index[corr['GDP'] == max_corr].tolist()[0]
        max_list.append(idx)
    df = pd.DataFrame(max_list, index = ['Italy', 'Japan', 'UAE', 'Canada', 'South Africa'], columns = ['Highest correlation variable'])
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

def adfuller_test(data):
  adf_test = adfuller(data, autolag='AIC') # AIC is the default option
  print('ADF Statistic:', adf_test[0])
  print('p-value: ', adf_test[1])
  print('Critical Values:')
  for key, value in adf_test[4].items():
      print('\t%s: %.3f' % (key, value))
  if adf_test[1] <= 0.05:
    print('We can reject the null hypothesis (H0) --> data is stationary')
  else:
    print('We cannot reject the null hypothesis (H0) --> data is non-stationary')

def kpss_test(data):
  kpss_out = kpss(data,regression='c', nlags='auto', store=True)
  print('KPSS Statistic:', kpss_out[0])
  print('p-value: ', kpss_out[1])
  if kpss_out[1] <= 0.05:
    print('We can reject the null hypothesis (H0) --> unit root, data is not stationary')
  else:
    print('We cannot reject the null hypothesis (H0) --> data is trend stationary')

def log_transform(df_train_test):
    for nation in df_train_test:
        df_train_test[nation] = (np.log(df_train_test[nation][0]), np.log(df_train_test[nation][1]))
    return df_train_test

def difference(df_train_test, difference):
    for nation in df_train_test:
        df_train_test[nation] = (df_train_test[nation][0].diff(difference).dropna(), df_train_test[nation][1].diff().dropna())
    return df_train_test