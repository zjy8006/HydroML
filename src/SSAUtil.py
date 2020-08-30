import pandas as pd
import os
root = os.path.dirname(os.path.abspath('__file__'))
import sys
sys.path.append(root)
from src.ssa import SSA

def ssa_decomposition(data,window):
    # print("window={}".format(window))
    columns=['ORIG']
    for i in range(window):
        columns.append("S"+str(i+1))
    decomposition = SSA(data,window)
    # print("decomposition:\n{}".format(decomposition))
    dec_df = decomposition.orig_TS
    for i in range(window):
        dec_df = pd.concat([dec_df,decomposition.reconstruct(i)],axis=1)
    dec_df = pd.DataFrame(dec_df.values,columns=columns)
    return dec_df