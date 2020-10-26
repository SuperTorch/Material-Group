# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:00:36 2020

@author: EDZ
"""

import pandas as pd
import numpy as np
import warnings
import woe_tools
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
# 数据加载

df = pd.read_excel('./material_group2.xlsx')

risk_cols = ['risk' + str(i) for i in range(1,25)]
money_cols = ['money' + str(i) for i in range(1,10)]
x_risk = df[risk_cols]
y_risk = df['risk_label']
x_money = df[money_cols]
y_money = df['money_label']

x_data = x_money.copy()
y_data = y_money.copy()
bin_cols = ['bin_money' + str(i) for i in range(1,10)]

feature_cols, score_card, model = woe_tools.work(x_data, y_data, bin_cols)

# 随机选择Good的5个人
good_sample = x_data[x_data['target'] == 0].sample(5)
good_sample = good_sample[feature_cols]
# 对5个好人进行评分
print(woe_tools.cal_score(good_sample, score_card))

# 随机选择Bad的5个人
bad_sample = x_data[x_data['target'] == 1].sample(5)
bad_sample = bad_sample[feature_cols]
# 对5个坏人进行评分
print(woe_tools.cal_score(bad_sample, score_card))