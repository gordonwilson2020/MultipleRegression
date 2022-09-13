####################################################################
# Multiple Regression Analysis with Python                         #
# Multiple Regression Assumptions                                  #
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

##########################################

# 2. Multiple Regression Analysis Data

# 2.1. Data Reading
data = pd.read_csv('Data//Multiple-Regression-Analysis-Data.txt', index_col='Date', parse_dates=True)

##########################################

# 3. Variables Definition

# 3.1. Squared Independent Variables
ivar = data[['t1y', 't10y', 'hyield', 'cpi', 'ppi', 'oil', 'indpro', 'pce']]
sivar = ivar ** 2
sivar.columns = ['st1y', 'st10y', 'shyield', 'scpi', 'sppi', 'soil', 'sindpro', 'spce']
data = data.join(sivar)

##########################################

# 4. Multiple Regression
data.loc[:, 'int'] = ct.add_constant(data)
lmivar = ['int', 't1y', 't10y', 'hyield', 'cpi', 'ppi', 'oil', 'indpro', 'pce']
lm = rg.OLS(data['stocks'], data[lmivar], hasconst=bool).fit()

##########################################

# 5. Multiple Regression Assumptions

# 5.1. Correct Specification
print('')
print('== Original Regression ==')
print('')
print(lm.summary())
print('')

# 5.1.1. Variable Selection (Step 1)
csivar1 = ['int', 't10y', 'hyield', 'cpi', 'ppi', 'oil', 'indpro', 'pce']
cslm1 = rg.OLS(data['stocks'], data[csivar1], hasconst=bool).fit()
print('== Variable Selection (Step 1) ==')
print('')
print(cslm1.summary())
print('')

# 5.1.2. Variable Selection (Step 2)
csivar2 = ['int', 't10y', 'hyield', 'cpi', 'oil', 'indpro', 'pce']
cslm2 = rg.OLS(data['stocks'], data[csivar2], hasconst=bool).fit()
print('== Variable Selection (Step 2) ==')
print('')
print(cslm2.summary())
print('')

# 5.1.3. Variable Selection (Step 3)
csivar3 = ['int', 't10y', 'hyield', 'cpi', 'oil', 'indpro']
cslm3 = rg.OLS(data['stocks'], data[csivar3], hasconst=bool).fit()
print('== Variable Selection (Step 3) ==')
print('')
print(cslm3.summary())
print('')

# 5.1.4. Variable Selection (Step 4)
csivar4 = ['int', 't10y', 'hyield', 'cpi', 'oil']
cslm4 = rg.OLS(data['stocks'], data[csivar4], hasconst=bool).fit()
print('== Variable Selection (Step 4) ==')
print('')
print(cslm4.summary())
print('')

# 5.1.5. Variable Selection (Step 5)
csivar5 = ['int', 't10y', 'hyield', 'oil']
cslm5 = rg.OLS(data['stocks'], data[csivar5], hasconst=bool).fit()
print('== Variable Selection (Step 5) ==')
print('')
print(cslm5.summary())
print('')

# 5.1.6. Variable Selection (Step 6)
csivar6 = ['int', 't10y', 'hyield']
cslm6 = rg.OLS(data['stocks'], data[csivar6], hasconst=bool).fit()
print('== Variable Selection (Step 6) ==')
print('')
print(cslm6.summary())
print('')

# 5.1.7. Variable Selection (Step 7)
csivar7 = ['int', 'hyield']
cslm7 = rg.OLS(data['stocks'], data[csivar7], hasconst=bool).fit()
print('== Variable Selection (Step 7) ==')
print('')
print(cslm7.summary())
print('')

# 5.2. No Linear Dependency
# Multi-collinearity Test
# No Multi-collinerity = Principal Diagonal Inverted Correlation Matrix < 10

# 5.2.1. Correlation Matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(ivar.corr(), cmap='Blues')
fig.colorbar(cax)
ax.set_xticklabels([''] + ['t1y', 't10y', 'hyield', 'cpi', 'ppi', 'oil', 'indpro', 'pce'])
ax.set_yticklabels([''] + ['t1y', 't10y', 'hyield', 'cpi', 'ppi', 'oil', 'indpro', 'pce'])
ax.set_title('Correlation Matrix')
plt.show()

# 5.2.2. Inverted Correlation Matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(np.linalg.inv(ivar.corr()), cmap='Greens')
fig.colorbar(cax)
ax.set_xticklabels([''] + ['t1y', 't10y', 'hyield', 'cpi', 'ppi', 'oil', 'indpro', 'pce'])
ax.set_yticklabels([''] + ['t1y', 't10y', 'hyield', 'cpi', 'ppi', 'oil', 'indpro', 'pce'])
ax.set_title('Inverted Correlation Matrix')
plt.show()

# 5.3. Correct Functional Form
# Ramsey-RESET Test
# Linearity (Ramsey-RESET test) = fittedvalues ** 2 p-value > 0.05

# 5.3.1. Linearity Ramsey-RESET Test
scslm7fv = cslm7.fittedvalues ** 2
data['scslm7fv'] = scslm7fv
cffivar = ['int', 'hyield', 'scslm7fv']
cfflm = rg.OLS(data['stocks'], data[cffivar], hasconst=bool).fit()
print('== Linearity Ramsey-RESET Test ==')
print('')
print(cfflm.summary())
print('')

# 5.4. Residuals No Auto-correlation
# Breusch-Godfrey Test
# No Auto-correlation (Breusch-Godfrey test 1) = residuals(-1) p-value > 0.05
# No Auto-correlation (Breusch-Godfrey test 2) = residuals(-1 to -12) F p-value > 0.05

# 5.4.1. Residuals No Auto-correlation Breusch-Godfrey (Test 1)
cslm7res = cslm7.resid
data['cslm7res'] = cslm7res
lcslm7res = cslm7res.shift(1)
lcslm7res = np.nan_to_num(lcslm7res)
data['lcslm7res'] = lcslm7res
rnaivar = ['int', 'hyield', 'lcslm7res']
rnalm = rg.OLS(data['cslm7res'], data[rnaivar], hasconst=bool).fit()
print('== Residuals No Auto-correlation Breusch-Godfrey (Test 1) ==')
print('')
print(rnalm.summary())
print('')

# 5.4.2. Residuals No Auto-correlarion Breusch-Godfrey (Test 2)
print('== Residuals No Auto-correlation Breusch-Godfrey (Test 2) ==')
print('')
print('Breusch-Godfrey LM Test (BG):', np.round(dg.acorr_breusch_godfrey(cslm7, nlags=12)[0], 6))
print('Prob (BG):', np.round(dg.acorr_breusch_godfrey(cslm7, nlags=12)[1], 6))
print('')

# 5.5. Residuals Homoscedasticity
# White and Breusch-Pagan Tests
# Homoscedasticity (White test) = F p-value > 0.05
# Homoscedasticity (Breusch-Pagan test) = F p-value > 0.05

# 5.5.1. Residuals Homoscedasticity White Test (No Cross Terms)
scslm7res = cslm7res ** 2
data['scslm7res'] = scslm7res
rhivar = ['int', 'hyield', 'shyield']
rhlm = rg.OLS(data['scslm7res'], data[rhivar], hasconst=bool).fit()
print('== Residuals Homoscedasticity White Test (No Cross Terms) ==')
print('')
print(rhlm.summary())
print('')

# 5.5.2. Residuals Homoscedasticity Breusch-Pagan Test
print('== Residuals Homoscedasticity Breusch-Pagan Test ==')
print('')
print('Breusch-Pagan LM Test (BP):', np.round(dg.het_breuschpagan(cslm7res, exog_het=data[csivar7])[0], 6))
print('Prob (BP):', np.round(dg.het_breuschpagan(cslm7res, exog_het=data[csivar7])[1], 6))
print('')

# 5.5.3. Heteroscedasticity Consistent Standard Errors
hccslm7 = rg.OLS(data['stocks'], data[csivar7], hasconst=bool).fit(cov_type='HC0')
print('== Original Regression ==')
print('')
print(cslm7.summary())
print('')
print('== Heteroscedasticity Consistent Standard Errors Regression ==')
print('')
print(hccslm7.summary())
print('')

# 5.6. Residuals Normality
# Jarque-Bera Test
# Normality (Jarque-Bera test) = residuals Jarque-Bera statistic < 5.99
print('== Residuals Normality Jarque-Bera Test==')
print('')
print('Jarque-Bera (JB):', np.round(jb.jarque_bera(cslm7res)[0], 6))
print('Prob (JB):', np.round(jb.jarque_bera(cslm7res)[1], 6))


