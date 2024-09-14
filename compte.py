import pandas as pd
import re
from sklearn.preprocessing import StandardScaler

to_normalize = ['credit_reel_estate', 'credit_consomation_depreciable', 'credit_consomation_renewable', 'outstanding_saving_account', 'term_saving_account', 'life_insurrance_account']

def _read_accounts():
    df = pd.read_csv('Synth_Compte.csv', encoding='ISO-8859-1', sep=';').rename(columns={
        'Période': 'period',
        'ID client': 'client_id',
        'Nombre de retraits effectués par carte': 'nb_card_withdrawal',
        "Nombre d'opérations carte": 'nb_card_operation',
        "Nombre d'opérations Hors carte": 'nb_noncard_operation',
        "Total mouvement d'affaires": 'total_movement',
        'Encours crédit immo': 'credit_reel_estate',
        "Encours Crédit consommation amortisable": 'credit_consomation_depreciable',
        'Encours Crédit consommation renouvelable': 'credit_consomation_renewable',
        'Encours Compte Epargne': 'outstanding_saving_account',
        'Encours Compte Epargne à terme': 'term_saving_account',
        'montant encours AV': 'life_insurrance_account'
    })
        
    for e in ('nb_noncard_operation', 'nb_card_operation', 'total_movement'):
        df[e] = df[e].map(lambda x: float( re.sub(r"\s+", "", x.replace(',', '')) ))

    for e in to_normalize:
        df[e] = df[e].map(lambda x: float(x.replace(',', '.').replace(' ', '')))
    
    scaler = StandardScaler()
    df[to_normalize] = scaler.fit_transform(df[to_normalize])
    return df, scaler

_cache, _scaler = _read_accounts()

def get_default():
    return [-float(m/s) for m, s in zip(_scaler.mean_, _scaler.scale_)]

def get_accounts():
    return _cache