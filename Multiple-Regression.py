####################################################################
# Multiple Regression Analysis with Python                         #
# Multiple Regression                                              #
# (c) Diego Fernandez Garcia 2015-2019                             #
# www.exfinsis.com                                                 #
####################################################################

# 1. Packages Importing
import numpy as np
import pandas as pd
import statsmodels.regression.linear_model as rg
import statsmodels.tools.tools as ct

##########################################

# 2. Multiple Regression Analysis Data

# 2.1. Data Reading
data = pd.read_csv('Data//Multiple-Regression-Analysis-Data.txt', index_col='Date', parse_dates=True)

##########################################

# 3. Multiple Regression
data.loc[:, 'int'] = ct.add_constant(data)
lmivar = ['int', 't1y', 't10y', 'hyield', 'cpi', 'ppi', 'oil', 'indpro', 'pce']
lm = rg.OLS(data['stocks'], data[lmivar], hasconst=bool).fit()
print('')
print(lm.summary())
print('')
print('== Analysis of Variance ANOVA ==')
print('')
print('== Degrees of Freedom DF ==')
print('regression degrees of freedom:', lm.df_model)
print('residuals degrees of freedom:', lm.df_resid)
print('total degrees of freedom:', lm.df_model + lm.df_resid)
print('')
print('== Sum of Squares SS ==')
print('regression sum of squares:', np.round(lm.ess, 6))
print('residuals sum of squares:', np.round(lm.ssr, 6))
print('total sum of squares:', np.round(lm.ess + lm.ssr, 6))
print('')
print('== Mean Square Error MSE ==')
print('regression mean square error:', np.round(lm.mse_model, 6))
print('residuals mean square error:', np.round(lm.mse_resid, 6))
print('total mean square error:', np.round(lm.mse_total, 6))
print('')
print('== F Test ==')
print('F-statistic:', np.round(lm.fvalue, 6))
print('Prob (F-statistic):', np.round(lm.f_pvalue, 6))


