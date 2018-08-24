#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import pandas as pd
from sklearn.linear_model import LinearRegression

#.fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)
def Temperature():
    # 讀取所需數據文件
    df_co2_data = pd.read_csv('CO2ppm.csv', header=0, index_col='Year')
    df_co2_data.index = pd.to_datetime(df_co2_data.index.astype(str))
    df_global_temp = pd.read_csv('GlobalSurfaceTemperature.csv', header=0, index_col='Year')
    df_global_temp.index = pd.to_datetime(df_global_temp.index.astype(str))
    df_ghg_data = pd.read_csv('GreenhouseGas.csv', header=0, index_col='Year')
    df_ghg_data.index = pd.to_datetime(df_ghg_data.index.astype(str))
    df_merge = pd.concat([df_ghg_data, df_co2_data, df_global_temp], axis=1)
    feature = df_merge.iloc[:, 0:4].fillna(method='ffill').fillna(method='bfill')
    feature_train = feature['1970-01-01':'2010-01-01']  # 訓練集
    feature_test = feature['2011-01-01':'2017-01-01']  # 要預測的目標

    # Median 预测
    target_Median = df_merge.iloc[:, 4]
    target_Median_train = target_Median['1970-01-01':'2010-01-01']
    model_Median = LinearRegression()
    model_Median.fit(feature_train, target_Median_train)
    MedianPredict = model_Median.predict(feature_test)

    # Upper
    target_Upper = df_merge.iloc[:, 5]
    target_Upper_train = target_Upper['1970-01-01':'2010-01-01']
    model_Upper = LinearRegression()
    model_Upper.fit(feature_train, target_Upper_train)
    UpperPredict = model_Upper.predict(feature_test)

    # Lower
    target_Lower = df_merge.iloc[:, 6]
    target_Lower_train = target_Lower['1970-01-01':'2010-01-01']
    model_Lower = LinearRegression()
    model_Lower.fit(feature_train, target_Lower_train)
    LowerPredict = model_Lower.predict(feature_test)

    '''
    補充代碼：
    1. 查看數據文件結構。
    2. 讀取數據並對缺失值處理
    3. 對時間序列數據集進行處理並重新採樣
    4. 整理數據
    5. 使用 scikit-learn 預測
    6. 將預測結果按列表返回
    '''

    # 將預測結果按 2011-2017 年份順序，並保留 3 位小數後以列表形成儲存

    # 按高、中、低依次返回預測結果列表
    return list(UpperPredict), list(MedianPredict), list(LowerPredict)
