import pandas as pd
import numpy as np
import bottleneck as bn
#from numba import jit
from typing import Iterable

def existNone(lst: Iterable) -> bool:
    return any(item is None for item in lst)


def abs(x: pd.DataFrame) -> pd.DataFrame:
    return np.abs(x)#x.abs()

def log(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x[x!=0])

def sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x)

def cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    return x.rank(axis=1, pct=True)

def delay(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.shift(d)

def correlation(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d//2).corr(y)#.replace([-np.inf, np.inf], 0).fillna(value=0)

def ts_corr_bn(df_1:pd.DataFrame,df_2:pd.DataFrame,window: int) -> pd.DataFrame:
    with np.errstate(all="ignore"):
        x = np.array(df_1)
        #x = df_1
        y = np.array(df_2)
        #y = df_2
        x = x + 0 * y
        y = y + 0 * x
        min_count = window//2
        mean_x_y = bn.move_mean(x*y, window=window, min_count=min_count,axis=0)
        mean_x = bn.move_mean(x, window=window, min_count=min_count,axis=0)
        mean_y = bn.move_mean(y, window=window, min_count=min_count,axis=0)
        count_x_y = bn.move_sum((np.isnan(x+y) == 0).astype(int), window=window, min_count=window//2,axis=0)
        x_var = bn.move_var(x, window=window, min_count=min_count,axis=0 , ddof = 1)
        y_var = bn.move_var(y, window=window, min_count=min_count,axis=0 , ddof = 1)

        numerator = (mean_x_y - mean_x * mean_y) * (
            count_x_y / (count_x_y - 1)
        )
        denominator = (x_var * y_var) ** 0.5
        result = numerator / denominator
    return pd.DataFrame(result,index = df_1.index,columns = df_1.columns)

def covariance(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d//2).cov(y)#.replace([-np.inf, np.inf], 0).fillna(value=0)

def cs_scale(x: pd.DataFrame, a:int=1) -> pd.DataFrame:
    return x.mul(a).div(x.abs().sum())

def ts_delta(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.diff(d)

def signedpower(x: pd.DataFrame, a:int) -> pd.DataFrame:
    return x**a


def ts_decay_linear_jit(x: pd.DataFrame, d:int) -> pd.DataFrame:
    # 過去 d 天的加權移動平均線，權重線性衰減 d, d ‒ 1, ..., 1（重新調整為總和為 1）
    @jit(nopython=True, nogil=True,cache = False,parallel=True)
    def con(result,x_array):
        for i in np.arange(1, 10):
            result[i:] += (i+1) * x_array[:-i]
        result[:d] = np.nan
        return result
    return pd.DataFrame(con(x.values.copy(),x.values) / np.arange(1, d+1).sum(),index = x.index,columns = x.columns)
def ts_decay_linear(x: pd.DataFrame, d:int) -> pd.DataFrame:
    # 過去 d 天的加權移動平均線，權重線性衰減 d, d ‒ 1, ..., 1（重新調整為總和為 1）
    result = x.values.copy()
    with np.errstate(all="ignore"):
        for i in range(1, d):
            result[i:] += (i+1) * x.values[:-i]
    result[:d] = np.nan
    return pd.DataFrame(result / np.arange(1, d+1).sum(),index = x.index,columns = x.columns)


def ts_min(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d//2).min()
def ts_min_bn(df:pd.DataFrame, window:int=10)->pd.DataFrame:
    return pd.DataFrame(bn.move_min(df, window=window, min_count=window//2,axis=0),columns = df.columns,index = df.index)


def ts_max(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d//2).max()
def ts_max_bn(df:pd.DataFrame, window:int=10)->pd.DataFrame:
    return pd.DataFrame(bn.move_max(df, window=window, min_count=window//2,axis=0),columns = df.columns,index = df.index)

def ts_argmin(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d//2).apply(np.nanargmin, raw=True)+1
def ts_argmin_bn(df:pd.DataFrame, window:int=10)->pd.DataFrame:
    deduction = np.array([range(1,df.shape[0]+1)]).T
    deduction[deduction > window]=window
    return pd.DataFrame(deduction - bn.move_argmin(df, window=window,min_count=window//2,axis=0),columns = df.columns,index = df.index)


def ts_argmax(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d//2).apply(np.nanargmax, raw=True)+1
def ts_argmax_bn(df:pd.DataFrame, window:int=10)->pd.DataFrame:
    deduction = np.array([range(1,df.shape[0]+1)]).T
    deduction[deduction > window]=window
    return pd.DataFrame(deduction - bn.move_argmax(df, window=window,min_count=window//2,axis=0),columns = df.columns,index = df.index)


def ts_rank(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d//2).rank(pct=True)

def min(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return np.minimum(x,y)

def max(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return np.maximum(x,y)

def ts_sum(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d//2).sum()
def ts_sum_bn(x:pd.DataFrame, d:int) -> pd.DataFrame:
    return pd.DataFrame(bn.move_sum(x, window=d, min_count=max(1,d//2),axis=0),columns = x.columns,index = x.index)

def ts_product(x: pd.DataFrame, d:int) -> pd.DataFrame:
    #return x.rolling(d, min_periods=d//2).apply(np.prod, raw=True)
    result = x.values.copy()
    with np.errstate(all="ignore"):
        for i in range(1, d):
            result[i:] *= x.values[:-i]
    return pd.DataFrame(result,index = x.index,columns = x.columns)

def ts_stddev(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d//2).std()
def ts_std_bn(df:pd.DataFrame, window:int=10)->pd.DataFrame:
    return pd.DataFrame(bn.move_std(df, window=window, min_count=window//2,axis=0 , ddof = 1),columns = df.columns,index = df.index)


def where(condition: pd.DataFrame, choiceA: pd.DataFrame, choiceB: pd.DataFrame) -> pd.DataFrame:
    condition_copy = pd.DataFrame(np.nan, index = condition.index, columns=condition.columns)
    condition_copy[condition] = choiceA
    condition_copy[~condition] = choiceB
    return condition_copy

def ts_mean(x: pd.DataFrame, d:int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d//2).mean()
def ts_mean_bn(x:pd.DataFrame, d:int) -> pd.DataFrame:
    return pd.DataFrame(bn.move_mean(x, window=d, min_count=d//2,axis=0),columns = x.columns,index = x.index)



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
    X.index.names = ['date','order_book_id']
    Neutralization_df = pd.concat([Y_stack,X],axis = 1).dropna().groupby('date').apply(lambda data:pd.DataFrame(get_rsid(data.iloc[:,1:].values,data.iloc[:,0].values),columns = ['rsid'],index = data.index.get_level_values('order_book_id')))['rsid']
    Neutralization_df.index.names = [None,'order_book_id']
    return Neutralization_df.unstack().reindex_like(Y)