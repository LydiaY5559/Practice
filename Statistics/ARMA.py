# -*- coding:utf8 -*-

import datetime
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

import common

start_date = '20150701'
end_date = '20160103'

def get_data():
	f = open('./data/ARMA_in_td18_income.txt', 'r')
	col1_data = []
	col2_data = []
	for row in f.readlines():
		col1, col2 = row.split(',')
		col1_data += [datetime.datetime.strptime(col1.strip(), '%Y-%m-%d')]
		col2_data += [float(col2.strip())]
	f.close()
	return pd.Series(col2_data, index=col1_data)

def save_data(data_series, filename):
	f = open('./'+filename, 'w')
	for i in data_series.index:
		f.write(i.strftime('%Y-%m-%d') + '\t' + str(data_series[i]) + '\n')
	f.close()

data = get_data()
ax1_1 = data.plot(figsize=(16,8))

fig2 = plt.figure(figsize=(12,8))
ax2_1 = fig2.add_subplot(211)
fig2 = sm.graphics.tsa.plot_acf(data.values.squeeze(), lags=40, ax=ax2_1)
ax2_2 = fig2.add_subplot(212)
fig2 = sm.graphics.tsa.plot_pacf(data, lags=40, ax=ax2_2)

# idx_list = np.array([])
# aic_list = np.array([])
# bic_list = np.array([])
# rsquared_list = np.array([])
# for order in range(1, 15):
# 	arma_mod = sm.tsa.ARMA(data, (order,0)).fit()
# 	predict_data = arma_mod.predict(start=start_date, end=end_date)
# 	idx_list = np.r_[idx_list, order]
# 	aic_list = np.r_[aic_list, arma_mod.aic]
# 	bic_list = np.r_[bic_list, arma_mod.bic]
# 	rsquared_list = np.r_[rsquared_list, common.r_squared(data.values.squeeze(), predict_data.values.squeeze())]
# fig3 = plt.figure(figsize=(12,8))
# ax3_1 = fig3.add_subplot(211)
# ax3_1.plot(idx_list, aic_list, label='aic')
# ax3_1.plot(idx_list, bic_list, label='bic')
# ax3_1.legend()
# ax3_2 = fig3.add_subplot(212)
# ax3_2.plot(idx_list, rsquared_list)

arma_mod = sm.tsa.ARMA(data, (7,0)).fit()
ax1_1 = arma_mod.plot_predict(start=start_date, end=end_date, ax=ax1_1)
predict_data = arma_mod.predict(start=start_date, end=end_date)
print arma_mod.summary()
print 'r_squared : '
print common.r_squared(data.values.squeeze(), predict_data.values.squeeze())

# arma_mod = sm.tsa.ARMA(data, (7,0)).fit()
# ax1_1 = arma_mod.plot_predict(start='20151201', end='20160701', ax=ax1_1)
# print arma_mod.summary()


# print 'sm.stats.durbin_watson(arma_mod.resid.values)'
# print sm.stats.durbin_watson(arma_mod.resid.values)

# fig3 = plt.figure(figsize=(12,8))
# ax3_1 = fig3.add_subplot(111)
# ax = arma_mod.resid.plot(ax=ax3_1)

# resid = arma_mod.resid
# print 'stats.normaltest(resid)'
# print stats.normaltest(resid)
# fig4 = plt.figure(figsize=(12,8))
# ax4_1 = fig4.add_subplot(111)
# fig4 = qqplot(resid, line='q', ax=ax4_1, fit=True)

# fig5 = plt.figure(figsize=(12,8))
# ax5_1 = fig5.add_subplot(211)
# fig5 = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax5_1)
# ax5_2 = fig5.add_subplot(212)
# fig5 = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax5_2)

# predict_data = arma_mod.predict(start=start_date, end=end_date)
# save_data(predict_data, './data/ARMA_out_td18_arppu.txt')

plt.show()