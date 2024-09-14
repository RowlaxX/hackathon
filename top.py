import pandas as pd

def _read_top():
    df = pd.read_csv('Top_Personne.csv', sep=';').rename(columns={
        'Période': 'period',
        'ID client': 'client_id',
        ' Top Prospect': 'top_prospect',
        'Top Client Actif': 'top_client_active',
        'Top Compte pro': 'top_account_pro',
        'Top Client contentieux': 'top_client_litigation',
        'Top Client recouvrement amiable': 'top_client_amicable_collecting',
        'Top Client en surendettement': 'top_client_overcharge',
        'Top Client droit au compte': 'top_client_account_right',
        'Top Client Interdit banque de france': 'top_client_bank_forbidden',
        'Top Client Incapable': 'top_client_unable',
        'Top Client en déshérence': 'top_client_deserhence',
        'Top Client en Fragilité': 'top_client_fragile'
    })
    return df

_cache = _read_top()

def get_tops():
    return _cache