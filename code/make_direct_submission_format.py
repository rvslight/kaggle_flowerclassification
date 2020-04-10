import pandas as pd
import numpy as np
import os

def make_direct_submit(output_file):
    df = pd.read_csv('submission.csv')
    df = df.astype(str)
    df['direct'] = df[['id','label']].apply(lambda x: ", ".join(x), axis=1) #axis column direction. apply each row.
    df.to_csv(output_file, index=False)
# df = pd.read_csv('submission.csv')
# df = df.astype(str)
# # print(df)
# df['new'] = df[['id','label']].apply(lambda x: ', '.join(x),axis=1) # 1 apply to each row..
# # df = df.set_index('id')
# print(df.head())
# sub = pd.merge(df['id'],df,on='id', how='left')
# print(sub.head())

make_direct_submit('direct_submission.csv')
