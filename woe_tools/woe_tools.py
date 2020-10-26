# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:27:42 2020

@author: cheng zixu
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

def get_bin_features(df, q=5):
    columns = df.columns
    for column in columns:
        df['bin_'+column] = pd.qcut(df[column], q=q, duplicates='drop')
    return df

# 计算IV，衡量自变量的预测能力
def cal_IV(df, feature, target):
    lst = []
    cols = ['feature', 'val', 'num', 'bad_num'] # 分别代表 字段名称、分箱数值段、在该分箱数值段的总个数、在该分箱数值中bad的个数
    for i in range(df[feature].nunique()): #nunique = unique的个数
        val = list(df[feature].unique())[i]
        # 统计feature，feature_value， 这个value的个数，这个value导致target为1的个数
        lst.append([feature, val])
        temp1 = df[df[feature]==val].count()[feature] # 这个value的总个数
        temp2 = df[(df[feature]==val) & (df[target]==1)].count()[feature] # 这个value导致target为1的个数
        #print(feature, val, temp1, temp2)
        lst.append([feature, val, temp1, temp2])
    data = pd.DataFrame(lst, columns=cols)
    data = data[data['bad_num']>0]
    data['num_ratio'] = data['num'] / data['num'].sum()
    data['bad_ratio'] = data['bad_num'] / data['num']
    data['bad_margin_rate'] = data['bad_num'] / data['bad_num'].sum()
    data['good_margin_rate'] = (data['num'] - data['bad_num'])/(data['num'].sum() - data['bad_num'].sum())
    data['WOE'] = np.log(data['bad_margin_rate']/data['good_margin_rate'])
    data['IV'] = data['WOE'] * (data['bad_margin_rate'] - data['good_margin_rate'])
    
    data = data.sort_values(by=['feature', 'val'])
    return data['IV'].sum()

# 计算df中bin字段的IV值， target为目标列
def cal_all_IV(df, bin_cols, target):
    # 统计所有字段的IV值
    result = {}
    for col in bin_cols:
        temp_IV = cal_IV(df, col, target)
        result[col] = temp_IV
    return result

def col_filtered(col_IVs, threshold=0.1):
    result = {k:v for k, v in col_IVs.items() if v >= threshold}
    return result.keys()

# 计算字段的WOE特征值
def cal_WOE(df, features, target):
    for feature in features:
        df_woe = df.groupby(feature).agg({target:['sum', 'count']})
        df_woe.columns = list(map(''.join, df_woe.columns.values))
        #print(df_woe.columns)
        df_woe = df_woe.reset_index()
        df_woe = df_woe.rename(columns={target+'sum': 'bad_num', target+'count': 'num'})
        #print(df_woe)
       
        df_woe['good_num'] = df_woe['num'] - df_woe['bad_num']
        df_woe = df_woe[[feature, 'good_num', 'bad_num']]
        
        df_woe['bad_margin_rate'] = df_woe['bad_num'] / df_woe['bad_num'].sum()
        df_woe['good_margin_rate'] = df_woe['good_num'] / df_woe['good_num'].sum()
        
        #计算woe
        df_woe['woe'] = np.log1p(df_woe['bad_margin_rate']/df_woe['good_margin_rate'])
        # 在后面拼接上 _feature, 比如_age
        df_woe.columns = [c if c == feature else c + '_' + feature for c in list(df_woe.columns.values)]
        # 拼接
        df = df.merge(df_woe, on=feature, how='left')
    return df

# 得到WOE规则
def get_woe_rules(df_woe, feature_cols):
    df_bin_to_woe = pd.DataFrame(columns=['features', 'bin', 'woe'])
    for f in feature_cols:
        b =  'bin_' + f 
        w = 'woe_bin_' + f
        df = df_woe[[w, b]].drop_duplicates()
        df.columns = ['woe', 'bin']
        df['features'] = f
        df = df[['features', 'bin', 'woe']]
        df_bin_to_woe = pd.concat([df_bin_to_woe, df])
    return df_bin_to_woe
    
def lr_model(df_woe, woe_cols, split, random_state):
    if split:
        X_train, X_test, y_train, y_test = train_test_split(df_woe[woe_cols], \
                                        df_woe['target'], test_size=0.2, random_state=34)
        model = LogisticRegression(random_state = random_state, class_weight='balanced', max_iter=500).fit(X_train, y_train)
        print('LR模型的准确率：',model.score(X_test,y_test))
        y_pred = model.predict(X_test)
        print('LR模型的F1 Score：',f1_score(y_pred, y_test))
        print('LR模型的AUC：',roc_auc_score(y_pred, y_test))
    else:
        X_train, y_train = df_woe[woe_cols], df_woe['target']
        model = LogisticRegression(random_state = random_state, class_weight='balanced', max_iter=500).fit(X_train, y_train)
    
    return model

def generate_scorecard(model_coef, binning_df, features, B):
    lst=[]
    cols = ['Variable', 'Binning', 'Score']
    # 模型系数
    coef = model_coef[0]
    for i in range(len(features)):
        f = features[i]
        # 得到这个feature的WOE规则
        df = binning_df[binning_df['features'] == f]
        for index, row in df.iterrows():
            lst.append([f, row['bin'], int(round(-coef[i] * row['woe'] * B))])
    data = pd.DataFrame(lst, columns=cols)
    return data

# 把数据映射到分箱中
def str2int(s):
    if s == '-inf':
        return -999999
    if s == 'inf':
        return 999999
    return float(s)

# 将value映射到bin
def map_value2bin(feature_value, feature2bin):
    for index, row in feature2bin.iterrows():
        bins = str(row['Binning'])
        left_open = bins[0]=='('
        right_open = bins[-1] == ')'
        binnings = bins[1:-1].split(',')
        in_range = True
        temp = str2int(binnings[0])
        temp2 = str2int(binnings[1])
        # 检查左括号
        if left_open:
            if feature_value <= temp:
                in_range = False
        else:
            if feature_value < temp:
                in_range = False
        # 检查右括号
        if right_open:
            if feature_value >= temp2:
                in_range = False
        else:
            if feature_value > temp2:
                in_range = False
        if in_range:
            return row['Binning']

# df为待转换样本， scorecard为评分卡规则
def map2score(df, score_card):
    scored_columns = list(score_card['Variable'].unique())
    score = 0
    for col in scored_columns:
        # 取出评分规则
        feature2bin = score_card[score_card['Variable'] == col]
        # 取出具体的feature_value
        feature_value = df[col]
        selected_bin = map_value2bin(feature_value, feature2bin)
        temp_score = feature2bin[feature2bin['Binning'] == selected_bin]
        score += temp_score['Score'].values[0]
    return score

def cal_score(df, score_card, A = 600):
    # map2score 按评分卡规则进行计算
    df['Score'] = df.apply(map2score, args=(score_card,), axis = 1)
    df['Score'] = df['Score'].astype(int)
    df['Score'] = df['Score'] + A
    return df

def work(x_data, y_data, bin_cols,split=True, random_state=33, B=20):
    #A = 600, B = 50
    x_data['target'] = y_data
    x_data = get_bin_features(x_data)
    col_IVs = cal_all_IV(x_data, bin_cols,'target')
    col_filter = col_filtered(col_IVs, 0.1)
    df_woe = cal_WOE(x_data, col_filter, 'target')
    woe_cols = [i for i in list(df_woe.columns.values) if 'woe' in i]
    feature_cols = [x[4:] for x in col_filter]
    df_bin_to_woe = get_woe_rules(df_woe, feature_cols)
    model = lr_model(df_woe, woe_cols, split, random_state)
    score_card = generate_scorecard(model.coef_, df_bin_to_woe, feature_cols, B)

    return feature_cols, score_card, model           