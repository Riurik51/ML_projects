import pandas as pd
import argparse
import sys
import json
import re
import numpy as np

def f_t_i_j(series, i, j):
    ret = np.copy(series.values)
    ret = np.roll(ret, i)
    ret[i:] = np.cumsum(ret[i:], dtype=float)
    ret[i + j:] = ret[i + j:] - ret[i:-j]
    ret = ret / j
    return pd.Series(ret)

def f_t_i_j_d(series, i, j, d):
    ret = np.copy(series.values).astype(float)
    ret = np.roll(ret, i)
    n = int(np.ceil((len(ret[i:])) / d))
    l = len(ret[i:])
    b = np.zeros(n * d)
    b[:l] = ret[i:]
    b = b.reshape(-1, d)
    b = np.cumsum(b, dtype=float, axis=0)
    b[j:] = (b[j:] - b[:-j]) / j
    ret[i:] = b.ravel()[:l]
    return pd.Series(ret)

def g_t_i_beta(series, i, beta):
    ans = np.array(series.shift(i).values, dtype=float)
    betas = beta ** (np.arange(len(ans) - i))
    ans[i:] = np.cumsum(ans[i:] * betas[::-1]) / betas[::-1]
    ans[i:] = ans[i:] / np.cumsum(betas)
    return pd.Series(ans)

def g_t_i_beta_d(series, i, beta, d):
    ans = np.array(series.shift(i).values, dtype=float)
    n = int(np.ceil((len(ans) - i) / d))
    b = np.zeros(n * d)
    b[:len(ans) - i] = ans[i:]
    b = b.reshape(-1, d)
    betas = beta ** (np.arange(n))
    b = np.cumsum(b.T * betas[::-1], axis=1) / betas[::-1]
    b = b / np.cumsum(betas)
    ans[i:] = b.T.ravel()[:len(ans) - i]
    return pd.Series(ans)

def createParser ():
    parser = argparse.ArgumentParser()
    parser.add_argument ('-c', '--config', type=str)
 
    return parser


def generate_past_demand_features(
        feature_df,
        shift_parameters,
        mean_parameters,
        mean_period_parameters,
        exp_mean_parameters,
        exp_mean_period_parameters,
):
    """
    
    Генерация признаков на основе предыдущих продаж

    Parameters
    ----------
    feature_df : DataFrame
        Датафрейм, полученный после скрипта generate_melted_df.py
    shift_parameters : list of tuples
    mean_parameters : list of tuples
    mean_period_parameters : list of tuples
    exp_mean_parameters : list of tuples
    exp_mean_period_parameters : list of tuples

    Examples
    --------
    Например для набора признаков
        f(t, 7, 1, 1), f(t, 28, 1, 1), f(t, 7, 7, 1), f(t, 28, 7, 1),
        f(t, 7, 7, 7), f(t, 28, 7, 7), g(t, 7, 0.5, 1), g(t, 7, 0.5, 7)
    аргументы будут следующими:
        shift_parameters=[7, 28]
        mean_parameters=[(7, 7), (28, 7)]
        mean_period_parameters=[(7, 7, 7), (28, 7, 7)]
        exp_mean_parameters=[(7, 0.5)]
        exp_mean_period_parameters=[(7, 0.5, 7)]
    
    Returns
    -------
    DataFrame
        Датафрейм с рассчитанными признаками
    Для ускорения просчета признаков и чтобы у нас не оставалось лишних NaN, делаю скользящее среднее по всем объектам без 		groupby - это даже выглядит логично заполнять значениями соседних id - это в большинстве случаев тот же товар в другом 		магазине или в соседней категории 
    """

    for shift_value in shift_parameters:
        column_name = f'shift_{shift_value}'
        feature_df[column_name] = np.roll(feature_df['demand'].values, shift_value)
        print("shift", shift_value)
  
    for shift_value, window_size in mean_parameters:
        column_name = f'mean_{shift_value}_{window_size}'
        feature_df[column_name] = f_t_i_j(feature_df['demand'], shift_value, window_size).values
        print("mean", shift_value, window_size)

    for shift_value, window_size, period_size in mean_period_parameters:
        column_name = f'mean_{shift_value}_{window_size}_period_{period_size}'
        feature_df[column_name] = f_t_i_j_d(feature_df['demand'], shift_value, window_size, period_size).values
        print("mean period", shift_value, window_size, period_size)

    for shift_value, beta in exp_mean_parameters:
        column_name = f'mean_{shift_value}_{beta:.3f}'
        feature_df[column_name] = feature_df.groupby('id').apply(lambda x: g_t_i_beta(x['demand'], shift_value, beta)).values
        print('exp ', shift_value, beta)

    for shift_value, beta, period_size in exp_mean_period_parameters:     
        column_name = f'mean_{shift_value}_{beta:.3f}_period_{period_size}'
        feature_df[column_name] = feature_df.groupby('id').apply(lambda x: g_t_i_beta_d(x['demand'], shift_value, beta, period_size)).values
        print('exp period', shift_value, beta, period_size)

    return feature_df
    
    
if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    config_string = re.sub('%', '"', namespace.config)
    config = json.loads(config_string)    

    id_columns = ['id']
    
    feature_df = pd.read_csv(
        'data/melted_train.csv',
        dtype={one_id_column: "category" for one_id_column in id_columns},
        usecols=id_columns + ['demand'],
    )
    
    feature_df = generate_past_demand_features(
        feature_df,
        **config,
    )
    
    # drop target
    feature_df.drop(['demand'], axis=1, inplace=True)

    feature_df.to_csv('data/past_demand_features.csv', index=False)
