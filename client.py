import pandas as pd
import utils

_csp = csp_dictionary = {
    "ARTISAN": [
        "ARTISAN REPARATIONS ET SERVICES", "ARTISAN ALIMENTATION", "ARTISAN BATIMENT", "ARTISAN FABRICATION"
    ],
    "CADRE": [
        "CADRE D'ENTREPRISE", "CHEF DE PETITE ENTREPRISE", "CHEF DE MOYENNE ENTREPRISE", 
        "CHEF DE GRANDE ENTREPRISE", "MEDECIN PHARMA SALARIES ETUD. HOSPI.", "MEDECIN SPECIALISTE LIBERAL", 
        "MEDECIN GENERALISTE LIBERAL", "PSYCH... (NON MEDECIN)", "CHIRURGIEN DENTISTE", "ARCHITECTE (LIBERAL)", 
        "VETERINAIRE", "PHARMACIEN LIBERAL", "AVOCAT, AVOUE", "NOTAIRE", "EXPERT COMPTA., COMPTA. AGREE (LIBERAL)", 
        "HUISSIER, OFFI. MINIS., DIV. PROF. LIB.", "CADRE FCT PUBLIQUE"
    ],
    "EMPLOYE": [
        "EMPLOYE DE COMMERCE", "EMPLOYE ADM. DES ENTREPRISES", "EMPLOYE FCT PUBLIQUE", 
        "PERS. DES SERV. DIRECTS AUX PARTICUL"
    ],
    "OUVRIER": [
        "OUVRIERS", "OUVRIERS AGRICOLES"
    ],
    "PROFESSION_INTERMEDIAIRE": [
        "TECHNIC. CONTREMAIT. AGT MAITRISE", "PROF. INTER. DES ENTREPRISES", "PROF. INTER. DE LA FCT PUBLIQUE", 
        "PROF. INTER. SANTE ET TRAV. SOCIAL", "PROFESSEUR ET ASSIMILE", "INSTITUTEUR ET ASSIMILE", 
        "CHERCHEUR FCT PUBLIQUE", "INFORMATION, ARTS, SPECTACLES", "CHEF DE CLINIQUE OU INTERNE DES HOPITAUX", 
        "AIDE FAMILIAL DE PROF. LIBERALE"
    ],
    "RETRAITE": [
        "RETRAITE (SANS PRECISION)", "ANCIEN EMPLOYE OUVRIER", "ANCIEN CADRE PROF. LIB. PROF. INTER.", 
        "ANCIEN ARTIS. COMMER. CHEF D'ENTR.",
        "ANCIEN ARTIS. COMMER. CHEF D'ENTR", "ANCIEN CADRE PROF. LIB. PROF. INTER",
    ],
    "ETUDIANT GRANDES ECOLES": [
        "ETUDIANT GRANDES ECOLES"
    ],
    "ETUDIANTS LIBERAL": [
        "ETUDIANT AUTRE QUE 8401 ET 8402", "ETUDIANT FUTUR PROFESSION LIBERALE",
    ],
    "SANS_ACTIVITE": [
        "SANS ACTIV. PROF. - 60 ANS", "SANS ACTIV. PROF. + 60 ANS", "CHOMEUR"
    ],
    "AGRICULTEUR": [
        "AGRICULTEUR PETITE EXPLOITATION"
    ],
    "AUTRES": [
        "PROFESSION INCONNUE", "CLERGE, RELIGIEUX"
    ],
    "ELEVE": [
        "ELEVE"
    ],
    "COMMERCANT": [
        "COMMERCANT ET ASSIMILE"
    ]
}

def _revertCsp():
    d = dict()
    for k, list in _csp.items():
        for e in list:
            d[e] = k
    return d

_cspReverted = _revertCsp()
print([(i, x) for i, x in enumerate(_csp.keys())])

def _read_clients() -> pd:
    df = pd.read_csv('Client.csv', encoding='ISO-8859-1', sep=';')
    df = df.rename(columns={
        'Période': 'period',
        'ID client': 'client_id',
        'AGE': 'age',
        'situation familiale': 'familial_situation',
        "date entrée en relation LCL": 'entry_date',
        'capacité juridique': 'juridic_capacity',
        'régime matrimonial': 'matrimonial_regime',
        "Nombre d'enfants": 'nb_kids',
        'Age enfant 1': 'age_kid_1',
        'Age enfant 2': 'age_kid_2',
        'Age enfant 3': 'age_kid_3',
        'Age enfant 4': 'age_kid_4',
        'Age enfant 5': 'age_kid_5',
        'CSP': 'job'
    })
    df = df.drop(columns=['entry_date'], axis=1)
    df['job'] = df['job'].map(lambda x: _cspReverted[x])
    df = utils.onehotencode(df, ['matrimonial_regime'])
    df = df.groupby(by=['period', 'client_id'], as_index=False).max()
    df = utils.onehotencode(df, ['juridic_capacity', 'familial_situation', 'job'])
    return df

_cache = _read_clients()

def get_clients() -> pd.DataFrame:
    return _cache