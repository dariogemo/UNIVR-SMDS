import pandas as pd
import numpy as np

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

def prova(corr_df):
    cgdpd = corr_df.drop('GDP', axis = 1).replace({1 : np.nan})
    idx = cgdpd.stack().idxmin()
    return idx