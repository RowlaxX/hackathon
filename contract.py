import pandas as pd

_cache = pd.read_csv('Synth_Contrat.csv', sep=';').rename(columns={
        'Période': 'period',
        'ID client': 'client_id',
        'ID contrat': 'contract_id',
        'libellé contrat': 'contract_type'
    })

def get_contracts():
    return _cache