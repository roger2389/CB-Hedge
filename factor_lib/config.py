import os
import Alpha101
DB_path = os.path.join(os.path.expanduser('~/Documents'),'MBQ_TW_DB')
data_pool = ['開盤價','收盤價','最高價','最低價','調整係數','成交金額_元','周轉率','當沖買賣占比','高低價差','資券互抵','資券互抵_元','本益比','董事總持股數']
base_factor_expr = ['Close', 'Low', 'High', 'High_Low_Diff','Adjust_Factor', 'Margin_Day_Trading_Amt', 'Day_Trading_Pct', 'Margin_Day_Trading_Vol', 'Total_Director_Holdings', 'Value_Dollars', 'Volume_1000_Shares', 'Turnover', 'Open', 'PER_TWSE','mktcap']
extra_expr = ['Adjust_Factor']
expr_dict = {
    # 基础资料
    **{factor_name:factor_name for factor_name in base_factor_expr},
    **{factor_name:factor_name for factor_name in extra_expr},
    'Volume':'Volume_1000_Shares * 1000',
    'Vwap':'Value_Dollars/Volume',
    # 基础因子
    'Adj_Open' : 'Adjust_Factor * Open',
    'Adj_Close' : 'Adjust_Factor * Close',
    'Adj_High' : 'Adjust_Factor * High',
    'Adj_Low' : 'Adjust_Factor * Low',

    'daily_ret':'(Adj_Close/Adj_Close.shift()-1)',
    'MoM5':'Adj_Close/Adj_Close.shift(5)-1',
    'MoM20':'Adj_Close/Adj_Close.shift(20)-1',
    'MoM252':'Adj_Close/Adj_Close.shift(252)-1',

    'daily_EMA5':'daily_ret.ewm(span=5).mean()',
    'daily_EMA20':'daily_ret.ewm(span=20).mean()',
    'daily_EMA252':'daily_ret.ewm(span=252).mean()',

    'daily_EMASTD5':'daily_ret.ewm(span=5).std()',
    'daily_EMASTD20':'daily_ret.ewm(span=20).std()',
    'daily_EMASTD252':'daily_ret.ewm(span=252).std()',

    'daily_EMA_Sharpe5':'daily_EMA5/daily_EMASTD5',
    'daily_EMA_Sharpe20':'daily_EMA20/daily_EMASTD20',
    'daily_EMA_Sharpe252':'daily_EMA252/daily_EMASTD252',

    'daily_EMA_Corr5':'daily_ret.ewm(span = 5, adjust=False).corr(daily_ret.mean(axis=1)).round(6)',
    'daily_EMA_Corr20':'daily_ret.ewm(span = 20, adjust=False).corr(daily_ret.mean(axis=1)).round(6)',
    'daily_EMA_Corr252':'daily_ret.ewm(span = 252, adjust=False).corr(daily_ret.mean(axis=1)).round(6)',

    'daily_EMA_Beta5':'daily_EMA_Corr5 * daily_EMASTD5.div(daily_ret.mean(axis=1).ewm(span=5).std(),axis = 0)',
    'daily_EMA_Beta20':'daily_EMA_Corr20 * daily_EMASTD20.div(daily_ret.mean(axis=1).ewm(span=20).std(),axis = 0)',
    'daily_EMA_Beta252':'daily_EMA_Corr252 * daily_EMASTD252.div(daily_ret.mean(axis=1).ewm(span=252).std(),axis = 0)',

    'intraday_ret':'(Adj_Close/Adj_Open-1)',
    'intraday_EMA5':'intraday_ret.ewm(span=5).mean()',
    'intraday_EMA20':'intraday_ret.ewm(span=20).mean()',
    'intraday_EMA252':'intraday_ret.ewm(span=252).mean()',

    'intraday_EMASTD5':'intraday_ret.ewm(span=5).std()',
    'intraday_EMASTD20':'intraday_ret.ewm(span=20).std()',
    'intraday_EMASTD252':'intraday_ret.ewm(span=252).std()',

    'intraday_EMA_Sharpe5':'intraday_EMA5/intraday_EMASTD5',
    'intraday_EMA_Sharpe20':'intraday_EMA20/intraday_EMASTD20',
    'intraday_EMA_Sharpe252':'intraday_EMA252/intraday_EMASTD252',

    'intraday_EMA_Corr5':'intraday_ret.ewm(span = 5, adjust=False).corr(intraday_ret.mean(axis=1)).round(6)',
    'intraday_EMA_Corr20':'intraday_ret.ewm(span = 20, adjust=False).corr(intraday_ret.mean(axis=1)).round(6)',
    'intraday_EMA_Corr252':'intraday_ret.ewm(span = 252, adjust=False).corr(intraday_ret.mean(axis=1)).round(6)',

    'intraday_EMA_Beta5':'intraday_EMA_Corr5 * intraday_EMASTD5.div(intraday_ret.mean(axis=1).ewm(span=5).std(),axis = 0)',
    'intraday_EMA_Beta20':'intraday_EMA_Corr20 * intraday_EMASTD20.div(intraday_ret.mean(axis=1).ewm(span=20).std(),axis = 0)',
    'intraday_EMA_Beta252':'intraday_EMA_Corr252 * intraday_EMASTD252.div(intraday_ret.mean(axis=1).ewm(span=252).std(),axis = 0)',

    'overnight_ret':'(Adj_Open/Adj_Close.shift(1)-1)',
    'overnight_EMA5':'overnight_ret.ewm(span=5).mean()',
    'overnight_EMA20':'overnight_ret.ewm(span=20).mean()',
    'overnight_EMA252':'overnight_ret.ewm(span=252).mean()',

    'overnight_EMASTD5':'overnight_ret.ewm(span=5).std()',
    'overnight_EMASTD20':'overnight_ret.ewm(span=20).std()',
    'overnight_EMASTD252':'overnight_ret.ewm(span=252).std()',

    'overnight_EMA_Sharpe5':'overnight_EMA5/overnight_EMASTD5',
    'overnight_EMA_Sharpe20':'overnight_EMA20/overnight_EMASTD20',
    'overnight_EMA_Sharpe252':'overnight_EMA252/overnight_EMASTD252',
    'overnight_EMA_Sharpe252_plus':'overnight_EMA252**2/overnight_EMASTD252',

    'overnight_EMA_Corr5':'overnight_ret.ewm(span = 5, adjust=False).corr(overnight_ret.mean(axis=1)).round(6)',
    'overnight_EMA_Corr20':'overnight_ret.ewm(span = 20, adjust=False).corr(overnight_ret.mean(axis=1)).round(6)',
    'overnight_EMA_Corr252':'overnight_ret.ewm(span = 252, adjust=False).corr(overnight_ret.mean(axis=1)).round(6)',

    'overnight_EMA_Beta5':'overnight_EMA_Corr5 * overnight_EMASTD5.div(overnight_ret.mean(axis=1).ewm(span=5).std(),axis = 0)',
    'overnight_EMA_Beta20':'overnight_EMA_Corr20 * overnight_EMASTD20.div(overnight_ret.mean(axis=1).ewm(span=20).std(),axis = 0)',
    'overnight_EMA_Beta252':'overnight_EMA_Corr252 * overnight_EMASTD252.div(overnight_ret.mean(axis=1).ewm(span=252).std(),axis = 0)',


    'daily_reverse_ret':'overnight_ret-intraday_ret',
    'daily_reverse_EMA5':'daily_reverse_ret.ewm(span=5).mean()',
    'daily_reverse_EMA20':'daily_reverse_ret.ewm(span=20).mean()',
    'daily_reverse_EMA252':'daily_reverse_ret.ewm(span=252).mean()',

    'daily_reverse_EMASTD5':'daily_reverse_ret.ewm(span=5).std()',
    'daily_reverse_EMASTD20':'daily_reverse_ret.ewm(span=20).std()',
    'daily_reverse_EMASTD252':'daily_reverse_ret.ewm(span=252).std()',

    'daily_reverse_EMA_Sharpe5':'daily_reverse_EMA5/daily_reverse_EMASTD5',
    'daily_reverse_EMA_Sharpe20':'daily_reverse_EMA20/daily_reverse_EMASTD20',
    'daily_reverse_EMA_Sharpe252':'daily_reverse_EMA252/daily_reverse_EMASTD252',

    'daily_reverse_EMA_Corr5':'daily_reverse_ret.ewm(span = 5, adjust=False).corr(daily_reverse_ret.mean(axis=1)).round(6)',
    'daily_reverse_EMA_Corr20':'daily_reverse_ret.ewm(span = 20, adjust=False).corr(daily_reverse_ret.mean(axis=1)).round(6)',
    'daily_reverse_EMA_Corr252':'daily_reverse_ret.ewm(span = 252, adjust=False).corr(daily_reverse_ret.mean(axis=1)).round(6)',

    'daily_reverse_EMA_Beta5':'daily_reverse_EMA_Corr5 * daily_reverse_EMASTD5.div(daily_reverse_ret.mean(axis=1).ewm(span=5).std(),axis = 0)',
    'daily_reverse_EMA_Beta20':'daily_reverse_EMA_Corr20 * daily_reverse_EMASTD20.div(daily_reverse_ret.mean(axis=1).ewm(span=20).std(),axis = 0)',
    'daily_reverse_EMA_Beta252':'daily_reverse_EMA_Corr252 * daily_reverse_EMASTD252.div(daily_reverse_ret.mean(axis=1).ewm(span=252).std(),axis = 0)',

    # 东吴金工换手率系列
    'Turn20':'Turnover.rolling(20).mean()',
    'PctTurn':'Turnover / Turnover.rolling(40).mean().shift(20) - 1',
    'PctTurn20':'PctTurn.rolling(20).mean()',
    '振幅':'(Adj_High - Adj_Low) / Adj_Close.shift()',
    '振幅换手率因子':'(PctTurn * 振幅).rolling(20).mean()',
    'STR':'Turnover.rolling(20).std()',
    'Vol20':'(Adj_Close / Adj_Close.shift() - 1).rolling(20).std()',
    'SCR':'(STR / (Turnover.rolling(40,min_periods = 1).std().shift(20)).shift() - 1)',
    'DRET':'(Adj_Close - Adj_Open)/Adj_Close.shift()',
    'FAC':'(Adj_High-Adj_Low)/Adj_Close.shift()',
    'PLUS':'(2 * Adj_Close - Adj_High - Adj_Low) / Adj_Close.shift()',
    
    'Turn20_mul_PLUS':'非負化處理(Zscore(Turn20)) * 非負化處理(Zscore(PLUS))',
    'STR_mul_PLUS':'非負化處理(Zscore(STR)) * 非負化處理(Zscore(PLUS))',
    'Turn_dePLUS':'Neutralization(Turnover,PLUS)',
    'Turn20_dePLUS':'Turn_dePLUS.rolling(20).mean()',

    'STR_dePLUS':'Turn_dePLUS.rolling(20).std()',
    'PLUS_deTurn':'Neutralization(PLUS,Turnover)',
    'PLUS_deTurn20':'PLUS_deTurn.rolling(20).mean()',
    'TPS':'非負化處理(Zscore(Turn20_dePLUS)) * 非負化處理(Zscore(PLUS_deTurn20))',
    'SPS':'非負化處理(Zscore(STR_dePLUS)) * 非負化處理(Zscore(PLUS_deTurn20))',

    'CCV':'Adj_Close.rolling(20).corr(Turnover)',
    'CDCV':'Adj_Close.diff().rolling(20).corr(Turnover)',
    'CCOV':'(Adj_Close-Adj_Open).rolling(20).corr(Turnover)',
    'COV':'(Adj_Open-Adj_Close.shift()).rolling(20).corr(Turnover.shift())',
    'RPV':'Zscore(CCOV) - Zscore(COV)',

    'RPV_deRet20':'Neutralization(RPV,MoM20)',
    'RPV_deTurn20':'Neutralization(RPV,Turn20)',
    'RPV_deTurn20_deRet20':'Neutralization(RPV,Turn20,MoM20)',
    '每日换手率变化率':'Turnover / Turnover.shift()-1',
    '每日换手率变化率_MA20':'每日换手率变化率.rolling(20).mean()',
    'GTR':'每日换手率变化率.rolling(20).std()',
    #风格因子
    '日内报酬变化量20日指数加权波动度':'intraday_ret.pct_change().ewm(span = 20).mean()',
    **Alpha101.alpha_dict,
}