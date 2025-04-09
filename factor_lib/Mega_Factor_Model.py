from joblib import Parallel, delayed, parallel_backend
from catboost import CatBoostRanker
from catboost import Pool
import pandas as pd
import numpy as np
import tqdm
import copy

class Model:
    def __init__(self,DB_Handler,factorname_list):
        self.DB_Handler = DB_Handler
        self.n_jobs = -1
        self.factorname_list = factorname_list
        self.insample_start_date = '2010-01-04'
        self.insample_end_date = '2017-12-31'
        self.val_start_date = '2018-01-01'
        self.val_end_date = '2019-12-31'
        self.cat_model_save_path = 'cat_model'
        self.index = None
    
    def Custon_condition(self,expr:str):
        db_handler_backup = copy.deepcopy(self.DB_Handler)
        self._con = eval(expr,db_handler_backup)
        del db_handler_backup
        if (self._con.sum(axis=1) > 1023).sum():
            print(f'警告，有数据过大({self._con.sum(axis=1).max()} > 1023)GPU无法支援！！！')

    @property
    def con(self):
        if not hasattr(self,'_con'):
            DB_Handler = self.DB_Handler
            
            基础门槛 = DB_Handler['Close'].notna()
            成交量门槛 = DB_Handler['Volume'].rolling(5).mean() > 500*1000
            非注意股票 = (~DB_Handler['atten_fg']).fillna(True) # 当天讯号
            非處置股票 = (~DB_Handler['disp_fg']).shift(-1).fillna(True) # 当天讯号
            #可當沖股票 = (~DB_Handler['sbadt_fg']).shift(-1).fillna(True) # 当天讯号
            #成交额门槛 = (DB_Handler['Value_Dollars']).rolling(5).mean() > 8000000
            #平均振幅门槛 = DB_Handler['振幅'].rolling(5,min_periods=1).mean()>=0.01
            #股价门槛 = DB_Handler['Close'] < 100

            self._con = 基础门槛 & 成交量门槛 & 非處置股票 & 非注意股票# & 可當沖股票
            #self._con = 基础门槛 & 成交量门槛 & 成交额门槛 & 平均振幅门槛 & 股价门槛  & 非處置股票 & 非注意股票# & 可當沖股票
            #self._con = 成交量门槛 & 非處置股票 & 非注意股票 & 可當沖股票
            if (self._con.sum(axis=1) > 1023).sum():
                raise ValueError(f'警告，有数据过大({self._con.sum(axis=1).max()} > 1023)GPU无法支援！！！')
            '''
            DB_Handler = self.DB_Handler
            con_1 = (DB_Handler['Value_Dollars']).rolling(5).mean()>=10000000
            con_2 = DB_Handler['Close'] < 100
            con_3 = (DB_Handler['Adj_Open'].shift(-1) / DB_Handler['Adj_Close'] - 1)<0.08
            con_3.iloc[-1] = True
            self._con = con_1 & con_2 & con_3
            '''
        return self._con
    
    @property
    def overnight_exp_ret(self):
        if not hasattr(self,'_overnight_exp_ret'):
            DB_Handler = self.DB_Handler
            self._overnight_exp_ret = (DB_Handler['Adj_Open'].shift(-2) / DB_Handler['Adj_Close'].shift(-1) - 1)[self.con]
        return self._overnight_exp_ret
    
    @property
    def intraday_exp_ret(self):
        if not hasattr(self,'_intraday_exp_ret'):
            DB_Handler = self.DB_Handler
            #停损标的 = (DB_Handler['Adj_High']/DB_Handler['Adj_Close'].shift()-1)>=0.09
            #停损报酬 = ((DB_Handler['Adj_Close'].shift()*1.09)/DB_Handler['Adj_Open']-1)
            #self._intraday_exp_ret = -1 * (DB_Handler['Adj_Close'] / DB_Handler['Adj_Open'] - 1).where(~停损标的,停损报酬).shift(-1)[self.con]
            self._intraday_exp_ret = -1 * (DB_Handler['Adj_Close'] / DB_Handler['Adj_Open'] - 1).shift(-1)[self.con]
        return self._intraday_exp_ret
    
    @property
    def exp_ret(self):
        if not hasattr(self,'_exp_ret'):
            self._exp_ret = (self.intraday_exp_ret + self.overnight_exp_ret)[self.con]
        return self._exp_ret

    @property
    def factor_df(self):
        if not hasattr(self,'_factor_df'):
            DB_Handler = self.DB_Handler
            factorname_list = self.factorname_list
            con = self.con
            if self.index is not None:
                index = con[con].loc[self.index].stack().index
            else:
                index = con[con].stack().index
            def get_factor_stack(data_name,con,DB_Handler):
                if isinstance(data_name,str):
                    if self.index is not None:
                        return DB_Handler[data_name][con].loc[self.index].stack().reindex(index = index)
                    else:
                        return DB_Handler[data_name][con].stack().reindex(index = index)
                if isinstance(data_name,pd.Series):
                    return data_name.reindex(index = index)
            with parallel_backend('loky', n_jobs=self.n_jobs):
                self._factor_df = pd.DataFrame(np.column_stack(Parallel()(delayed(get_factor_stack)(data_name,con,DB_Handler) for data_name in tqdm.tqdm(factorname_list))),index=index,columns=factorname_list)
        return self._factor_df
    
    def train_cat(self,verbose:bool = True,task_type:str = 'GPU',**args):
        print('准备cat需求数据')
        start_date = self.insample_start_date
        end_date = self.insample_end_date
        target = (self.exp_ret[self.con].rank(axis=1, pct=True) * 100).round().stack().loc[f'{start_date}':f'{end_date}'].astype(int)
        X = self.factor_df.loc[target.index].dropna(how='all')
        y = target.loc[X.index]
        group_id = y.index.get_level_values(0).to_series().index.astype('category').codes

        val_start_date = self.val_start_date
        val_end_date = self.val_end_date
        val_target = (self.exp_ret[self.con].rank(axis=1, pct=True) * 100).round().stack().loc[f'{val_start_date}':f'{val_end_date}'].astype(int)
        val_X = self.factor_df.loc[val_target.index].dropna(how='all')
        val_y = val_target.loc[val_X.index]
        val_group_id = val_y.index.get_level_values(0).to_series().index.astype('category').codes

        print('开始训练cat')
        # 初始化 CatBoostRanker 模型
        model = CatBoostRanker(
            task_type=task_type,
            verbose=verbose,
            use_best_model=True,
            **args,
        )
        # 训练模型
        model.fit(Pool(data = X,label = y,group_id = group_id),
                  eval_set=Pool(data = val_X,label = val_y,group_id = val_group_id),  
                  early_stopping_rounds=50,)  # 連續 50 次迭代未改善就停止
        print('cat存档')
        model.save_model(self.cat_model_save_path)
    
    @property
    def cat_model(self):
        if not hasattr(self,'_loaded_ranker'):
            # 從文件加載模型
            loaded_ranker = CatBoostRanker()
            loaded_ranker.load_model(self.cat_model_save_path)
            self._loaded_ranker = loaded_ranker
            print("模型已加載。")
        return self._loaded_ranker
    
    def predict(self,factor_df:pd.DataFrame = None)->pd.DataFrame:
        # 使用加載的模型進行預測
        if factor_df is None:
            factor_df = self.factor_df
        y_pred_loaded = self.cat_model.predict(factor_df)
        print("加載模型的預測完成。")
        return pd.Series(y_pred_loaded,index =  factor_df.index).unstack()
    @property
    def IC_Se(self):
        if not hasattr(self,'_IC_Se'):
            self._IC_Se = self.predict().corrwith(self.exp_ret,axis = 1,method='spearman').sort_index()
        return self._IC_Se

    def IC_test(self):
        # 简单检定
        IC_Se = self.IC_Se
        insample_start_date = pd.to_datetime(self.insample_start_date)
        insample_end_date = pd.to_datetime(self.insample_end_date)
        IC_mean = pd.Series({
            f"样本内({insample_start_date.date()}~{insample_end_date.date()})":IC_Se.loc[insample_start_date:insample_end_date].mean(),
            f"样本外({(insample_start_date+pd.Timedelta(days=1)).date()}~{IC_Se.index[-1].date()})":IC_Se.loc[insample_start_date+pd.Timedelta(days=1):].mean(),
        })
        ICIR = pd.Series({
            f"样本内({insample_start_date.date()}~{insample_end_date.date()})":IC_Se.loc[insample_start_date:insample_end_date].mean() / IC_Se.loc[insample_start_date:insample_end_date].std(),
            f"样本外({(insample_start_date+pd.Timedelta(days=1)).date()}~{IC_Se.index[-1].date()})":IC_Se.loc[insample_start_date+pd.Timedelta(days=1):].mean() / IC_Se.loc[insample_start_date+pd.Timedelta(days=1):].std(),
        })
        return pd.concat({"IC":IC_mean,"ICIR":ICIR,},axis = 1)