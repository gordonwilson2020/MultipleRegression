####################################################################
# Multiple Regression Analysis with Python                         #
# Multiple Regression Forecasting                                  #
# (c) Diego Fernandez Garcia 2015-2019                             #
# www.exfinsis.com                                                 #
####################################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.regression.linear_model as rg
import statsmodels.tools.tools as ct
import statsmodels.stats.diagnostic as dg
import statsmodels.stats.api as jb
import statsmodels.tools.eval_measures as fm

##########################################

# 2. Multiple Regression Analysis Data

# 2.1. Data Reading
data = pd.read_csv('Data//Multiple-Regression-Analysis-Data.txt', index_col='Date', parse_dates=True)

##########################################

# 3. Variables Definition

# 3.1. Lagged Independent Variables
ivar = data[['t1y', 't10y', 'hyield', 'cpi', 'ppi', 'oil', 'indpro', 'pce']]
livar = ivar.shift(1)
livar.columns = ['lt1y', 'lt10y', 'lhyield', 'lcpi', 'lppi', 'loil', 'lindpro', 'lpce']
data = data.join(livar)

# 3.2. Squared Independent Variables
sivar = ivar.join(livar) ** 2
sivar.columns = ['st1y', 'st10y', 'shyield', 'scpi', 'sppi', 'soil', 'sindpro', 'spce',
                 'slt1y', 'slt10y', 'slhyield', 'slcpi', 'slppi', 'sloil', 'slindpro', 'slpce']
data = data.join(sivar)

# 3.3. Training and Testing Ranges Delimiting
data.loc[:, 'int'] = ct.add_constant(data)
tdata = data[:'2012-12-31'].dropna()
fdata = data['2013-01-31':]

##########################################

# 4. Multiple Regression (Training Range)
lmivart = ['int', 'lt1y', 'lt10y', 'lhyield', 'lcpi', 'lppi', 'loil', 'lindpro', 'lpce']
lmt = rg.OLS(tdata['stocks'], tdata[lmivart], hasconst=bool).fit()

##########################################

# 5. Multiple Regression Assumptions (Training Range)

# 5.1. Correct Specification (Training Range)
print('')
print('== Original Regression (Training Range) ==')
print('')
print(lmt.summary())
print('')

# 5.1.1. Variable Selection (Step 1) (Training Range)
csivart1 = ['int', 'lt1y', 'lt10y', 'lhyield', 'lcpi', 'lppi', 'lindpro', 'lpce']
cslmt1 = rg.OLS(tdata['stocks'], tdata[csivart1], hasconst=bool).fit()
print('== Variable Selection (Step 1) (Training Range) ==')
print('')
print(cslmt1.summary())
print('')

# 5.1.2. Variable Selection (Step 2) (Training Range)
csivart2 = ['int', 'lt1y', 'lt10y', 'lcpi', 'lppi', 'lindpro', 'lpce']
cslmt2 = rg.OLS(tdata['stocks'], tdata[csivart2], hasconst=bool).fit()
print('== Variable Selection (Step 2) (Training Range) ==')
print('')
print(cslmt2.summary())
print('')

# 5.1.3. Variable Selection (Step 3) (Training Range)
csivart3 = ['int', 'lt10y', 'lcpi', 'lppi', 'lindpro', 'lpce']
cslmt3 = rg.OLS(tdata['stocks'], tdata[csivart3], hasconst=bool).fit()
print('== Variable Selection (Step 3) (Training Range) ==')
print('')
print(cslmt3.summary())
print('')

# 5.1.4. Variable Selection (Step 4) (Training Range)
csivart4 = ['int', 'lt10y', 'lcpi', 'lindpro', 'lpce']
cslmt4 = rg.OLS(tdata['stocks'], tdata[csivart4], hasconst=bool).fit()
print('== Variable Selection (Step 4) (Training Range) ==')
print('')
print(cslmt4.summary())
print('')

# 5.1.5. Variable Selection (Step 5) (Training Range)
csivart5 = ['int', 'lt10y', 'lindpro', 'lpce']
cslmt5 = rg.OLS(tdata['stocks'], tdata[csivart5], hasconst=bool).fit()
print('== Variable Selection (Step 5) (Training Range) ==')
print('')
print(cslmt5.summary())
print('')

# 5.1.6. Variable Selection (Step 6) (Training Range)
csivart6 = ['int', 'lindpro', 'lpce']
cslmt6 = rg.OLS(tdata['stocks'], tdata[csivart6], hasconst=bool).fit()
print('== Variable Selection (Step 6) (Training Range) ==')
print('')
print(cslmt6.summary())
print('')

# 5.1.7. Variable Selection (Step 7) (Training Range)
csivart7 = ['int', 'lindpro']
cslmt7 = rg.OLS(tdata['stocks'], tdata[csivart7], hasconst=bool).fit()
print('== Variable Selection (Step 7) (Training Range) ==')
print('')
print(cslmt7.summary())
print('')

# 5.2. Correct Functional Form (Training Range)
# Ramsey-RESET Test
# Linearity (Ramsey-RESET test) = fittedvalues ** 2 p-value > 0.05

# 5.2.1. Linearity Ramsey-RESET (Test 1) (Training Range)
scslmt7fv = cslmt7.fittedvalues ** 2
tdata['scslmt7fv'] = scslmt7fv
cffivart1 = ['int', 'lindpro', 'scslmt7fv']
cfflmt1 = rg.OLS(tdata['stocks'], tdata[cffivart1], hasconst=bool).fit()
print('== Linearity Ramsey-RESET (Test 1) (Training Range) ==')
print('')
print(cfflmt1.summary())
print('')

# 5.2.2. Non-Linear Functional Form (Training Range)
nlivart = ['int', 'lindpro', 'slindpro']
nllmt = rg.OLS(tdata['stocks'], tdata[nlivart], hasconst=bool).fit()
print('== Non-Linear Functional Form Regression (Training Range) ==')
print('')
print(nllmt.summary())
print('')

# 5.2.3. Linearity Ramsey-RESET (Test 2) (Training Range)
snllmtfv = nllmt.fittedvalues ** 2
tdata['snllmtfv'] = snllmtfv
cffivart2 = ['int', 'lindpro', 'slindpro', 'snllmtfv']
cfflmt2 = rg.OLS(tdata['stocks'], tdata[cffivart2], hasconst=bool).fit()
print('== Linearity Ramsey-RESET (Test 2) (Training Range) ==')
print('')
print(cfflmt2.summary())
print('')

# 5.3. Residuals No Auto-correlation (Training Range)
# Breusch-Godfrey Test
# No Auto-correlation (Breusch-Godfrey test 1) = residuals(-1) p-value > 0.05
# No Auto-correlation (Breusch-Godfrey test 2) = residuals(-1 to -12) F p-value > 0.05

# 5.3.1. Residuals No Auto-correlation Breusch-Godfrey (Test 1) (Training Range)
nllmtres = nllmt.resid
tdata['nllmtres'] = nllmtres
lnllmtres = nllmtres.shift(1)
lnllmtres = np.nan_to_num(lnllmtres)
tdata['lnllmtres'] = lnllmtres
rnaivart = ['int', 'lindpro', 'slindpro', 'lnllmtres']
rnalmt = rg.OLS(tdata['nllmtres'], tdata[rnaivart], hasconst=bool).fit()
print('== Residuals No Auto-correlation Breusch-Godfrey (Test 1) (Training Range) ==')
print('')
print(rnalmt.summary())
print('')

# 5.3.2. Residuals No Auto-correlarion Breusch-Godfrey (Test 2) (Training Range)
print('== Residuals No Auto-correlation Breusch-Godfrey (Test 2) (Training Range) ==')
print('')
print('Breusch-Godfrey LM Test (BG):', np.round(dg.acorr_breusch_godfrey(nllmt, nlags=12)[0], 6))
print('Prob (BG):', np.round(dg.acorr_breusch_godfrey(nllmt, nlags=12)[1], 6))
print('')

# 5.4. Residuals Homoscedasticity (Training Range)
# Breusch-Pagan Test
# Homoscedasticity (Breusch-Pagan test) = F p-value > 0.05

# 5.4.1. Residuals Homoscedasticity Breusch-Pagan Test (Training Range)
print('== Residuals Homoscedasticity Breusch-Pagan Test (Training Range) ==')
print('')
print('Breusch-Pagan LM Test (BP):', np.round(dg.het_breuschpagan(nllmtres, exog_het=tdata[nlivart])[0], 6))
print('Prob (BP):', np.round(dg.het_breuschpagan(nllmtres, exog_het=tdata[nlivart])[1], 6))
print('')

# 5.5. Residuals Normality (Training Range)
# Jarque-Bera Test
# Normality (Jarque-Bera test) = residuals Jarque-Bera statistic < 5.99
print('== Residuals Normality Jarque-Bera Test (Training Range) ==')
print('')
print('Jarque-Bera (JB):', np.round(jb.jarque_bera(nllmtres)[0], 6))
print('Prob (JB):', np.round(jb.jarque_bera(nllmtres)[1], 6))
print('')

##########################################

# 6. Multiple Regression Forecasting (Testing Range)

# 6.1. Multiple Regression Forecasting Fitted Values (Testing Range)
nllmtfvf = nllmt.predict(fdata[nlivart])

# 6.2. Multiple Regression Forecasting Accuracy Metrics (Testing Range)
lstocksf = data['stocks'].shift(1)['2013-01-31':]
stockstmf = pd.Series(tdata['stocks'].mean()).repeat(len(fdata))
print('== Forecasting Accuracy Metrics (Testing Range) ==')
print('')
print('== Mean Absolute Error (Testing Range) ==')
print('Model:', np.round(fm.meanabs(nllmtfvf, fdata['stocks']), 6))
print('Random Walk:', np.round(fm.meanabs(lstocksf, fdata['stocks']), 6))
print('Arithmetic Mean:', np.round(fm.meanabs(stockstmf, fdata['stocks']), 6))
print('')
print('== Mean Squared Error (Testing Range) ==')
print('Model:', np.round(fm.mse(nllmtfvf, fdata['stocks']), 6))
print('Random Walk:', np.round(fm.mse(lstocksf, fdata['stocks']), 6))
print('Arithmetic Mean:', np.round(fm.mse(stockstmf, fdata['stocks']), 6))
print('')
print('== Root Mean Squared Error (Testing Range) ==')
print('Model:', np.round(fm.rmse(nllmtfvf, fdata['stocks']), 6))
print('Random Walk:', np.round(fm.rmse(lstocksf, fdata['stocks']), 6))
print('Arithmetic Mean:', np.round(fm.rmse(stockstmf, fdata['stocks']), 6))
print('')

# 6.3. Multiple Regression Forecasting Chart (Testing Range)
plt.plot(fdata['stocks'], label='stocks')
plt.plot(nllmtfvf, label='nllmtfvf')
plt.plot(lstocksf, label='lstocksf')
plt.plot(pd.DataFrame(stockstmf).set_index(fdata.index), label='stockstmf')
plt.title('Multiple Regression Forecasting Chart')
plt.legend(loc='upper left')
plt.show()

