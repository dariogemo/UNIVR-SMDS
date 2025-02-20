# Final project for my university class of Statistical Models for Data Science

The provided dataset contains estimates of total GDP (gross domestic product) and its components for several countries over the period 1970-2020. This can inform on how these measures have changed over time, and allows to investigate the usefulness of GDP as a measure of wellbeing. These information are available for 220 countries and 17 different indicators were derived.

In "project.ipynb" you can find the main notebook where the task is carried out. The code here is kept to a minimum to better understand the analysis of the time series and the results; for this purpose, a separate file called "functions.py" storing all the main functions run by the notebook has been created. For better clarity, some documentation about the functions is provided.

When run, the main notebook will ask a list of 5 countries that the user want to analyze and predict GDP for the years 2010-2020. A complete set of the available nations to input can be found in the file [valid_nations.csv](https://github.com/dariogemo/UNIVR-SMDS/blob/main/valid_nations.csv), which only contains nations from the original dataset that don't have null values and don't lack any of the 17 macroeconomic factors. 

If the user doesn't give any input, the notebook will use the following list as the default nations to analyze: Finland, Sweden, Portugal, Algeria, Germany. The GDP components that are used are construction, final consumption, government consumption and gross capital.

For every nation, the notebook follows the following pipeline. Firstly some exporatory data analysis is performed on the datasets. Then, 6 models are created that are later used to predict the GDP values from 2010 to 2020, and lastly the results are confronted to retrieve the best model. In the end the user should end up with the best models at predicting the GDP of the selected nations.

The models are the following:
1. SLR: simple linear regression with GDP as a dependent variable and time as as the predictor
2. MLR: multiple linear regression, adding the other macroeconomic indicators as predictors
3. ETS: a grid search is performed to find out the best parameters for the model, i.e error, trend, seasonal, seasonal_period and damped_trend.
4. ARIMAX: auto_arima function retrieves the best Arima model with exogenous variables that are all the remaining columns of the dataframe.
5. VARX: firstly the columns that granger cause GDP are retrieved. These columns, in addition to the GDP, are used to train the model. The remaining columns are set as exogenous variables. A grid search is performed to find the best "p" parameters.
6. VAR: on the contrary from the VARX, here only the best indicator is used as endogenous variable (with GDP) and no exogenous variables are provided. A grid search is performed to find the best "p" parameters.

The results of the best models for the default countries are the following.

![](results.png)