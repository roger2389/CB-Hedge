import pandas as pd
import numpy as np
import bottleneck as bn
# from numba import jit
from typing import Iterable

def existNone(lst: Iterable) -> bool:
    return any(item is None for item in lst)
# 自身处理
def abs(x: pd.DataFrame) -> pd.DataFrame:
    return np.abs(x)#x.abs()

def log(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x[x!=0])

def sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x)

def cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    return x.rank(axis=1, pct=True)

def cs_scale(x: pd.DataFrame, a:int=1) -> pd.DataFrame:
    return x.mul(a).div(x.abs().sum(axis = 1), axis='index')

def signedpower(x: pd.DataFrame, a:int) -> pd.DataFrame:
    return x**a

# 时序处理
def delay(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.shift(d)

def correlation(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d).corr(y)#.replace([-np.inf, np.inf], 0).fillna(value=0)

def ts_corr_bn(df_1:pd.DataFrame,df_2:pd.DataFrame,window: int) -> pd.DataFrame:
    with np.errstate(all="ignore"):
        x = np.array(df_1)
        #x = df_1
        y = np.array(df_2)
        #y = df_2
        x = x + 0 * y
        y = y + 0 * x
        #min_count = window//2
        mean_x_y = bn.move_mean(x*y, window=window,axis=0)
        mean_x = bn.move_mean(x, window=window,axis=0)
        mean_y = bn.move_mean(y, window=window,axis=0)
        count_x_y = bn.move_sum((np.isnan(x+y) == 0).astype(int), window=window,axis=0)
        x_var = bn.move_var(x, window=window,axis=0 , ddof = 1)
        y_var = bn.move_var(y, window=window,axis=0 , ddof = 1)

        numerator = (mean_x_y - mean_x * mean_y) * (
            count_x_y / (count_x_y - 1)
        )
        denominator = (x_var * y_var) ** 0.5
        result = numerator / denominator
    return pd.DataFrame(result,index = df_1.index,columns = df_1.columns)

def covariance(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d).cov(y)#.replace([-np.inf, np.inf], 0).fillna(value=0)

def ts_delta(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.diff(d)


# def ts_decay_linear_jit(x: pd.DataFrame, d:int) -> pd.DataFrame:
#     # 過去 d 天的加權移動平均線，權重線性衰減 d, d ‒ 1, ..., 1（重新調整為總和為 1）
#     @jit(nopython=True, nogil=True,cache = False,parallel=True)
#     def con(result,x_array):
#         for i in np.arange(1, 10):
#             result[i:] += (i+1) * x_array[:-i]
#         result[:d] = np.nan
#         return result
#     return pd.DataFrame(con(x.values.copy(),x.values) / np.arange(1, d+1).sum(),index = x.index,columns = x.columns)
def ts_decay_linear(x: pd.DataFrame, d:int) -> pd.DataFrame:
    # 過去 d 天的加權移動平均線，權重線性衰減 d, d ‒ 1, ..., 1（重新調整為總和為 1）
    result = x.values.copy().astype(float)
    with np.errstate(all="ignore"):
        for i in range(1, d):
            result[i:] += (i+1) * x.values[:-i]
    result[:d] = np.nan
    return pd.DataFrame(result / np.arange(1, d+1).sum(),index = x.index,columns = x.columns)


def ts_min(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).min()
def ts_min_bn(df:pd.DataFrame, window:int=10)->pd.DataFrame:
    return pd.DataFrame(bn.move_min(df, window=window,axis=0),columns = df.columns,index = df.index)


def ts_max(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).max()
def ts_max_bn(df:pd.DataFrame, window:int=10)->pd.DataFrame:
    return pd.DataFrame(bn.move_max(df, window=window,axis=0),columns = df.columns,index = df.index)

def ts_argmin(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).apply(np.nanargmin, raw=True)+1
def ts_argmin_bn(df:pd.DataFrame, window:int=10)->pd.DataFrame:
    deduction = np.array([range(1,df.shape[0]+1)]).T
    deduction[deduction > window]=window
    return pd.DataFrame(deduction - bn.move_argmin(df, window=window,axis=0),columns = df.columns,index = df.index)


def ts_argmax(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).apply(np.nanargmax, raw=True)+1
def ts_argmax_bn(df:pd.DataFrame, window:int=10)->pd.DataFrame:
    deduction = np.array([range(1,df.shape[0]+1)]).T
    deduction[deduction > window]=window
    return pd.DataFrame(deduction - bn.move_argmax(df, window=window,axis=0),columns = df.columns,index = df.index)


def ts_rank(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).rank(pct=True)

def ts_sum(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).sum()
def ts_sum_bn(x:pd.DataFrame, d:int) -> pd.DataFrame:
    return pd.DataFrame(bn.move_sum(x, window=d,axis=0),columns = x.columns,index = x.index)

def ts_product(x: pd.DataFrame, d:int) -> pd.DataFrame:
    #return x.rolling(d, min_periods=d//2).apply(np.prod, raw=True)
    result = x.values.copy()
    with np.errstate(all="ignore"):
        for i in range(1, d):
            result[i:] *= x.values[:-i]
    return pd.DataFrame(result,index = x.index,columns = x.columns)

def ts_stddev(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).std()
def ts_std_bn(df:pd.DataFrame, window:int=10)->pd.DataFrame:
    return pd.DataFrame(bn.move_std(df, window=window,axis=0 , ddof = 1),columns = df.columns,index = df.index)
def ts_mean(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d).mean()
def ts_mean_bn(x:pd.DataFrame, d:int) -> pd.DataFrame:
    return pd.DataFrame(bn.move_mean(x, window=d,axis=0),columns = x.columns,index = x.index)

# 两两处理
def min(*x_list) -> pd.DataFrame:
    if len(x_list)==2:
        return np.minimum(x_list[0],x_list[1])
    else:
        x = x_list[0].copy()
        for _ in x_list[1:]:
            x = np.minimum(x,_)
        return x
    
def max(*x_list) -> pd.DataFrame:
    if len(x_list)==2:
        return np.maximum(x_list[0],x_list[1])
    else:
        x = x_list[0].copy()
        for _ in x_list[1:]:
            x = np.maximum(x,_)
        return x

def add(*x_list) -> pd.DataFrame:
    if len(x_list)==2:
        return x_list[0] + x_list[1]
    else:
        return pd.concat(x_list, keys=range(len(x_list))).groupby(level=1).sum()

def mean(*x_list) -> pd.DataFrame:
    return pd.concat(x_list, keys=range(len(x_list))).groupby(level=1).mean()

def std(*x_list) -> pd.DataFrame:
    return pd.concat(x_list, keys=range(len(x_list))).groupby(level=1).std()

def sub(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return x-y

def mul(*x_list) -> pd.DataFrame:
    if len(x_list)==2:
        return x_list[0] * x_list[1]
    else:
        x = x_list[0].copy()
        for _ in x_list[1:]:
            x*=_
        return x

def truediv(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return x / (y + 1e-9)  # 避免除以零

def where(condition: pd.DataFrame, choiceA: pd.DataFrame, choiceB: pd.DataFrame) -> pd.DataFrame:
    condition_copy = pd.DataFrame(np.nan, index = condition.index, columns=condition.columns)
    condition_copy[condition] = choiceA
    condition_copy[~condition] = choiceB
    return condition_copy



Zscore = lambda df:(df.sub(df.mean(axis = 1),axis = 0)).div(df.std(axis = 1),axis = 0)
非負化處理 = lambda df:df.add(df.min(axis = 1).abs(),axis = 0)
def Neutralization(Y:pd.DataFrame,*X_list:pd.DataFrame):
    #中性化
    def get_rsid(X:np.ndarray,Y:np.ndarray):
        def get_beta(X:np.ndarray,Y:np.ndarray):
            # 添加截距项
            #X = np.column_stack((np.ones(X.shape[0]), X))
            # 计算回归系数
            coefficients = np.linalg.pinv(X.T @ X) @ X.T @ Y
            return coefficients
        coefficients = get_beta(X,Y)
        # 计算预测值
        predicted_Y = X @ coefficients
        # 计算残差（中性化后的因子值）
        rsid = Y - predicted_Y
        return rsid
    Y_stack = Y.stack()
    Y_stack.index.names = ['date','order_book_id']
    try:
        X = pd.concat({_+1:X_list[_] for _ in range(len(X_list))},axis = 1).stack(future_stack=True)
    except:
        X = pd.concat({_+1:X_list[_] for _ in range(len(X_list))},axis = 1).stack()
    try:
        X.index.names = ['date','order_book_id']
        Neutralization_df = pd.concat([Y_stack,X],axis = 1).dropna().groupby('date').apply(lambda data:pd.DataFrame(get_rsid(data.iloc[:,1:].values,data.iloc[:,0].values),columns = ['rsid'],index = data.index.get_level_values('order_book_id')))['rsid']
        Neutralization_df.index.names = [None,'order_book_id']
        return Neutralization_df.unstack().reindex_like(Y)
    except:
        return pd.DataFrame(np.nan,index=Y.index,columns=Y.columns)
def fast_Neutralization(Y:pd.DataFrame,X:pd.DataFrame):
    Beta = Y.corrwith(X,axis=1) * Y.std(axis=1) / X.std(axis=1)
    Alpha = Y - X.mul(Beta,axis=0)
    return Alpha
def Factor_to_weight(factor:pd.DataFrame,only_long:bool = False):
    demeaned = factor - np.nanmean(factor,axis = 1)[:, None]
    weights = demeaned / np.nansum(np.abs(demeaned),axis = 1)[:, None]
    weights[np.isnan(weights)] = 0#检查数据无值时视为权重0
    if only_long:
        weights[weights<0] = 0
        weights*=2
    return weights

Alpha_F = {"abs": abs,
                "correlation": ts_corr_bn,
                "covariance": covariance,
                "cs_rank": cs_rank,
                "cs_scale": cs_scale,
                "delay": delay,
                "log": log,
                "max": max,
                "min": min,
                "sign": sign,
                "signedpower": signedpower,
                "ts_argmax": ts_argmax_bn,
                "ts_argmin": ts_argmin_bn,
                "ts_decay_linear": ts_decay_linear,
                "ts_delta": ts_delta,
                "ts_max": ts_max_bn,
                "ts_min": ts_min_bn,
                "ts_product": ts_product,
                "ts_rank": ts_rank,
                "ts_stddev": ts_std_bn,
                "ts_sum": ts_sum_bn,
                "where": where,
                "ts_mean": ts_mean_bn,
                'add':add,
                'sub':sub,
                'mul':mul,
                'truediv':truediv,
                'Zscore':Zscore,
                '非負化處理':非負化處理,
                'fast_Neutralization':fast_Neutralization,
                'Neutralization':Neutralization,
                'Factor_to_weight':Factor_to_weight,
                'mean':mean,
                'std':std,
                }

