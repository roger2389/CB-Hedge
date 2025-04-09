import tqdm
import config
import Tool

import sys
import os
MBQ_tej_v2_path ="C:\Users\User\我的雲端硬碟 (owen.lin@mutual-boost.com)\MBQ_Tej_v2\MBQ_tej_v2"
if not os.path.exists(MBQ_tej_v2_path):
    MBQ_tej_v2_path = "/Users/ureychen/Library/CloudStorage/GoogleDrive-elijah.chen@mutual-boost.com/.shortcut-targets-by-id/1hn3mJXCct6IDRoNt3x63QSDz3RkPXCCg/MBQ_Tej_v2/MBQ_tej_v2"
if not os.path.exists(MBQ_tej_v2_path):
    MBQ_tej_v2_path  = r'G:\.shortcut-targets-by-id\1hn3mJXCct6IDRoNt3x63QSDz3RkPXCCg\MBQ_Tej_v2\MBQ_Tej_v2'
if not os.path.exists(MBQ_tej_v2_path):
    MBQ_tej_v2_path  = '/home/ubuntu/MBQ/MBQ_Tej_v2/MBQ_tej_v2'
sys.path.append(MBQ_tej_v2_path)
import MBQ_tej_v2_manager
import MBQ_risk_model

MBQ_tej_v2_Handler = MBQ_tej_v2_manager.Handler()
Handler = Tool.Handler()
for factor_name,expr in tqdm.tqdm(config.expr_dict.items()):
    if factor_name in config.base_factor_expr+config.extra_expr:
        if factor_name in ['atten_fg','disp_fg','sbadt_fg']:
            Handler[factor_name] = MBQ_tej_v2_Handler[factor_name] == 'Y'
        else:
            Handler[factor_name] = MBQ_tej_v2_Handler[factor_name]
    else:    
        Handler[factor_name] = eval(expr,Handler)
'''
import Alpha101
from joblib import Parallel, delayed
def save_data(factor_name,expr,Handler):
    Handler[factor_name] = eval(expr,Handler)
r = Parallel(n_jobs=-1)(delayed(save_data)(factor_name,expr,Handler) for factor_name,expr in tqdm.tqdm(Alpha101.alpha_dict.items()))'
'''