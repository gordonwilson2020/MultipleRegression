####################################################################
# Multiple Regression Analysis with Python                         #
# Variables Definition                                             #
# (c) Diego Fernandez Garcia 2015-2019                             #
# www.exfinsis.com                                                 #
####################################################################

# 1. Packages Importing
import pandas as pd
import matplotlib.pyplot as plt

##########################################

# 2. Multiple Regression Analysis Data

# 2.1. Data Reading
data = pd.read_csv('Data//Multiple-Regression-Analysis-Data.txt', index_col='Date', parse_dates=True)

##########################################

# 3. Variables Definition

# 3.1. Dependent Variable

# stocks Dependent Variable
print('== stocks Dependent Variable ==')
print(data['stocks'].describe())
print('')

plt.plot(data['stocks'], label='stocks')
plt.title('stocks Dependent Variable')
plt.legend(loc='upper left')
plt.show()

# 3.2. Independent Variables

# 3.2.1. Rates Independent Variables

# t1y Independent Variable
print('== t1y Independent Variable ==')
print(data['t1y'].describe())
print('')

plt.plot(data['t1y'], label='t1y')
plt.title('t1y Independent Variable')
plt.legend(loc='upper left')
plt.show()

# t10y Independent Variable
print('== t10y Independent Variable ==')
print(data['t10y'].describe())
print('')

plt.plot(data['t10y'], label='t10y')
plt.title('t10y Independent Variable')
plt.legend(loc='upper left')
plt.show()

# hyield Independent Variable
print('== hyield Independent Variable ==')
print(data['hyield'].describe())
print('')

plt.plot(data['hyield'], label='hyield')
plt.title('hyield Independent Variable')
plt.legend(loc='upper left')
plt.show()

# 3.2.2. Prices Independent Variables

# cpi Independent Variable
print('== cpi Independent Variable ==')
print(data['cpi'].describe())
print('')

plt.plot(data['cpi'], label='cpi')
plt.title('cpi Independent Variable')
plt.legend(loc='upper left')
plt.show()

# ppi Independent Variable
print('== ppi Independent Variable ==')
print(data['ppi'].describe())
print('')

plt.plot(data['ppi'], label='ppi')
plt.title('ppi Independent Variable')
plt.legend(loc='upper left')
plt.show()

# oil Independent Variable
print('== oil Independent Variable ==')
print(data['oil'].describe())
print('')

plt.plot(data['oil'], label='oil')
plt.title('oil Independent Variable')
plt.legend(loc='upper left')
plt.show()

# 3.2.3. Macroeconomic Independent Variables

# indpro Independent Variable
print('== indpro Independent Variable ==')
print(data['indpro'].describe())
print('')

plt.plot(data['indpro'], label='indpro')
plt.title('indpro Independent Variable')
plt.legend(loc='upper left')
plt.show()

# pce Independent Variable
print('== pce Independent Variable ==')
print(data['pce'].describe())
print('')

plt.plot(data['pce'], label='pce')
plt.title('pce Independent Variable')
plt.legend(loc='upper left')
plt.show()

# 3.3. Variables Descriptive Statistics

print('== Variables Descriptive Statistics ==')
print('')
print('==  Mean ==')
print(data.mean())
print('')
print('== Standard Deviation ==')
print(data.std())
print('')
print('== Skewness ==')
print(data.skew())
print('')
print('== Excess Kurtosis ==')
print(data.kurt())
print('')

# 3.4. Lagged Independent Variables
ivar = data[['t1y', 't10y', 'hyield', 'cpi', 'ppi', 'oil', 'indpro', 'pce']]
livar = ivar.shift(1)
livar.columns = ['lt1y', 'lt10y', 'lhyield', 'lcpi', 'lppi', 'loil', 'lindpro', 'lpce']
data = data.join(livar)

# 3.5. Squared Independent Variables
sivar = ivar.join(livar) ** 2
sivar.columns = ['st1y', 'st10y', 'shyield', 'scpi', 'sppi', 'soil', 'sindpro', 'spce',
                 'slt1y', 'slt10y', 'slhyield', 'slcpi', 'slppi', 'sloil', 'slindpro', 'slpce']
data = data.join(sivar)
print(data[['t1y', 'lt1y', 'st1y', 'slt1y']].head())

# 3.6. Training and Testing Ranges Delimiting
tdata = data[:'2012-12-31'].dropna()
fdata = data['2013-01-31':]

