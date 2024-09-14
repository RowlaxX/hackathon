import pandas as pd
import utils

def _read_navigation():
    df = pd.read_csv('Tracefinal4yes.csv', sep=',').rename(columns={
        'ID client': 'client_id',
        'merged': 'page',
        'mois': 'period'
    })

    df = df.drop(columns=['Unnamed: 0'], axis=1)
    df = utils.onehotencode(df, ['page'])
    df = df.groupby(by=['client_id', 'period'], as_index=False).max()

    return df

_cache = _read_navigation()

def get_navigations():
    return _cache
