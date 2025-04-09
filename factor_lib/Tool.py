import os
import pandas as pd
import numpy as np
import tqdm
from joblib import Parallel, delayed
import time
import plotly.express as px
import os
import config
config.DB_path = os.path.join(os.path.expanduser('~/Documents'),'MBQ_TW_DB')
import quantstats as qs
qs.extend_pandas()
import cufflinks
cufflinks.go_offline()
import utils
import requests
import operators_v4

class Handler(dict):
    def __init__(self,path = config.DB_path,data_type:str = 'parquet'):
        self.path = path
        self.cashe_dict = {}
        self.func_dict = operators_v4.Alpha_F
        if data_type == 'pickle':
            data_type = 'pkl'
        self.data_type = data_type

        os.makedirs(path,exist_ok=True)
    def __getitem__(self, key):
        if key in self.cashe_dict:
            return self.cashe_dict[key]
        elif key in self.func_dict:
            return self.func_dict[key]
        else:
            file_path = os.path.join(self.path, f'{key}.{self.data_type}')
            # 检查存储的文件是否存在
            if os.path.exists(file_path):
                try:
                    if self.data_type == 'parquet':
                        return pd.read_parquet(file_path)
                    elif self.data_type == 'pkl':
                        return pd.read_pickle(file_path)
                except :
                    # 如果文件损坏或无法读取，返回默认值
                    raise ValueError(f'文件{file_path}损坏或无法读取')
            raise ValueError(f'文件{file_path}不存在')
    def __call__(self, key):
        return self.__getitem__(key)
    def __setitem__(self, key, value):
        file_path = os.path.join(self.path, f'{key}.{self.data_type}')
        value[np.isfinite(value)].to_parquet(file_path)
    def cash_list(self):
        parquet_set = set(filter(lambda X:X.endswith(f".{self.data_type}"),os.listdir(self.path)))
        return sorted(map(lambda X:X[:-(len(self.data_type)+1)],list(parquet_set)))

def get_mkts_exposure(Common_Stock):
    mkts_exposure = pd.Series(1,index = Common_Stock.set_index(['mdate','coid','mkt']).index).unstack().fillna(0)
    return mkts_exposure

def get_inds_exposure(Common_Stock):
    inds_exposure = pd.Series(1,index = Common_Stock.set_index(['mdate','coid','main_ind_c']).index).unstack().fillna(0)
    inds_exposure.columns = inds_exposure.columns.str.split(' ').str[-1]
    inds_exposure = inds_exposure.rename(columns = {'其他':'其它',
                                                    '':'其它',
                                                    '證券類':'証券',
                                                    '化學生技醫療':'化學生技',
                                                    '玻璃陶瓷':'玻璃',
                                                    '橡膠類':'橡膠工業',
                                                    '電子類':'電子工業',
                                                    '百貨類':'百貨',
                                                    '貿易百貨':'百貨',
                                                    })
    inds_exposure = inds_exposure.groupby(inds_exposure.columns,axis='columns').sum()
    return inds_exposure
def Factor_to_weight(factor:pd.DataFrame,only_long:bool = False):
    demeaned = factor - np.nanmean(factor,axis = 1)[:, None]
    weights = demeaned / np.nansum(np.abs(demeaned),axis = 1)[:, None]
    weights[np.isnan(weights)] = 0#检查数据无值时视为权重0
    if only_long:
        weights[weights<0] = 0
        weights*=2
    return weights
def max_drawdown(prices):
    # 計算累計的最大值
    cumulative_max = prices.cummax()
    # 計算回撤 (Drawdown)
    drawdown = (prices - cumulative_max) / cumulative_max
    # 計算最大回撤 (MDD)
    mdd = drawdown.min()
    return mdd

def 最大權重限制器(W_df:pd.DataFrame,最大權重門檻:float = 1/3)->pd.DataFrame:
    超過門檻的標的_bool = W_df>最大權重門檻
    W_df[超過門檻的標的_bool] = 最大權重門檻
    need_up_great_weight = W_df[~超過門檻的標的_bool].sum(axis=1)
    need_up_great_weight = need_up_great_weight.where(need_up_great_weight==1,(1-W_df[超過門檻的標的_bool].sum(axis=1))/need_up_great_weight)
    need_up_great_weight = need_up_great_weight[np.isfinite(need_up_great_weight)].reindex_like(need_up_great_weight)
    W_df[~超過門檻的標的_bool] = W_df[~超過門檻的標的_bool].mul(need_up_great_weight,axis=0)
    return W_df

def show_pie(w_df):
    df = w_df.iloc[-1][w_df.iloc[-1]>0].reset_index()
    df.columns = ['Category', 'Weight']
    # 使用 plotly express 
    fig = px.pie(df, values='Weight', names='Category', title='Weight Distribution',
                width=400, height=400,  
                labels={'Category': 'Category'},  
                hole=0.1)  
    fig.show()

def show_stats(bt_ret:pd.DataFrame)->None:
    if isinstance(bt_ret,pd.Series):
        bt_ret = pd.DataFrame({'策略':bt_ret})
    try:
        display(pd.concat({"CAGR(%)":bt_ret.cagr()*100,
                'Sharpe':bt_ret.mean()/bt_ret.std()*252**0.5,
                'Calmar':bt_ret.calmar(),
                'MDD(%)':bt_ret.max_drawdown()*100,
                '單利MDD(%)' : max_drawdown(bt_ret.cumsum().add(1))*100,
                '样本胜率(%)':bt_ret.apply(lambda X:((X.dropna()>0).sum()  / X.dropna().shape[0])*100),
                '周胜率(%)':bt_ret.apply(lambda X:((X.dropna().add(1).resample('W').prod().sub(1)>0).sum()  / X.dropna().add(1).resample('W').prod().sub(1).dropna().shape[0])*100),
                '月胜率(%)':bt_ret.apply(lambda X:((X.dropna().add(1).resample('ME').prod().sub(1)>0).sum()  / X.dropna().add(1).resample('ME').prod().sub(1).shape[0])*100),
                '年胜率(%)':bt_ret.apply(lambda X:((X.dropna().add(1).resample('YE').prod().sub(1)>0).sum()  / X.dropna().add(1).resample('YE').prod().sub(1).shape[0])*100),
                '盈亏比(avg_win/avg_loss)': bt_ret.apply(lambda X:(X[X > 0].mean() / abs(X[X < 0].mean()))),
                '总赚赔比(profit_factor)':bt_ret.profit_factor(),
                '预期报酬(bps)':((1 + bt_ret).prod() ** (1 / len(bt_ret)) - 1)*10000,
                '样本数':bt_ret.apply(lambda X:X.dropna().count()),
                },axis = 1).round(2))
    except:
        display(pd.concat({"CAGR(%)":bt_ret.cagr()*100,
                'Sharpe':bt_ret.mean()/bt_ret.std()*252**0.5,
                'MDD(%)':bt_ret.max_drawdown()*100,
                '單利MDD(%)' : max_drawdown(bt_ret.cumsum().add(1))*100,
                '样本胜率(%)':bt_ret.apply(lambda X:((X.dropna()>0).sum()  / X.dropna().shape[0])*100),
                '周胜率(%)':bt_ret.apply(lambda X:((X.dropna().add(1).resample('W').prod().sub(1)>0).sum()  / X.dropna().add(1).resample('W').prod().sub(1).dropna().shape[0])*100),
                '月胜率(%)':bt_ret.apply(lambda X:((X.dropna().add(1).resample('M').prod().sub(1)>0).sum()  / X.dropna().add(1).resample('M').prod().sub(1).shape[0])*100),
                '年胜率(%)':bt_ret.apply(lambda X:((X.dropna().add(1).resample('Y').prod().sub(1)>0).sum()  / X.dropna().add(1).resample('Y').prod().sub(1).shape[0])*100),
                '盈亏比(avg_win/avg_loss)': bt_ret.apply(lambda X:(X[X > 0].mean() / abs(X[X < 0].mean()))),
                '总赚赔比(profit_factor)':bt_ret.profit_factor(),
                '预期报酬(bps)':((1 + bt_ret).prod() ** (1 / len(bt_ret)) - 1)*10000,
                '样本数':bt_ret.apply(lambda X:X.dropna().count()),
                },axis = 1).round(2))
def backtest_factor(factor:pd.DataFrame,exp_ret:pd.DataFrame,rank_range_n:int = 10,start_date:str = '2019-01-01'):
    factor_rank = factor.rank(axis = 1,pct = True,method = 'first')

    IC_Se = factor.corrwith(exp_ret,axis=1,method='spearman').sort_index().loc[start_date:]
    print(f'IC_mean:{round(IC_Se.mean(),4)}')
    print(f'IC_IR:{round(IC_Se.mean()/IC_Se.std(),4)}')

    bt = pd.concat({f'{int(((_/rank_range_n)*100))}% ~ {int((_+1)/rank_range_n*100)}%':exp_ret[(factor_rank>_/rank_range_n) & (factor_rank<=(_+1)/rank_range_n)].mean(axis = 1) - exp_ret.mean(axis=1) for _ in tqdm.tqdm(range(rank_range_n))}, axis = 1).dropna(how = 'all')
    bt = bt.loc[start_date:]
    if (bt.iloc[:,-1] - bt.iloc[:,0]).add(1).prod() > 1:
        bt['LS_ret'] = bt.iloc[:,-1] - bt.iloc[:,0]
    else:
        bt['LS_ret'] = bt.iloc[:,0] - bt.iloc[:,-1]
    show_stats(bt)

    (bt.drop(columns='LS_ret').loc[start_date:].cagr()*100).iplot(kind = 'bar')
    bt.index = bt.index.astype(str)
    bt.cumsum().ffill().iplot()
    bt.index = pd.to_datetime(bt.index)

#'''

def get_one_USstock_EODHD(ticker: str, start_date: str = '1990-01-01', end_date: str = f'{pd.Timestamp.today().date()}') -> pd.DataFrame:
    for _ in range(5):
        try:
            # 在每个子进程中初始化 APIClient
            api_key = "67ced072980439.50625412"  # 你的 API 密钥
            #client = APIClient(api_key)  # 创建 APIClient 实例

            # 获取历史数据
            '''
            data = client.get_eod_historical_stock_market_data(
                symbol=f"{ticker}.US",  # 格式为 {股票代码}.{交易所}
                period="d",             # 日线数据
                from_date=start_date,   # 开始日期
                to_date=end_date,       # 结束日期
                order="a"               # 按日期升序排列
            )
            '''
            data = requests.get(f"https://eodhd.com/api/eod/ {ticker}.US ?from= {start_date} &to= {end_date} &period= d &api_token={api_key} &fmt=json").json()
            if len(data) == 0:
                return None
            # 将数据转换为 DataFrame
            df = pd.DataFrame(data)
            # 将日期列设置为索引
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 调整列名并计算调整后的价格
            df.rename(columns={'adjusted_close': 'adj_close'}, inplace=True)
            Adj_rate = df['adj_close'] / df['close']  # 计算调整因子
            for column in ['open', 'high', 'low']:
                df['adj_' + column] = Adj_rate * df[column]  # 计算调整后的价格
            df = df.sort_index(axis='columns')  # 按列名排序后返回
            return df
        except :
            print(f'抓取:{ticker} 时遇到问题 20秒后重试 ！！！')
            time.sleep(20)
    ValueError(f'抓取:{ticker} 时遇到问题！！！')
#'''