import contract
import compte
import client
import top
import navigation

Y_filter = {
    "Livret A",
    "Carte Cleo",
    "Livret Dév. Durable et Solidaire",
    "Carte Premier",
    "AMP FORMULE ETUDIANTE",
    "PRÊT IMMOBILIER",
    "LCL ETUDIANT",
    "ASSURANCE HABITATION",
    "Prêt Personnel Etudes",
    "LCL VIE ET PREVOYANCE",
    "ASSURANCE DECOUVERT AUTORISE HF",
    "LCL VIE JEUNES",
    "ASSURANCE EMPRUNTEUR LCL",
    "CAPITAL DECES",
    "Livret épargne populaire",
    "Livret Jeune",
    "Prêt Personnel Budget",
    "ASSURANCE REVENUS",
    "ASSURANCE AUTOMOBILE",
    "Prêt Personnel Auto",
    "Plan d'épargne logement",
    "Plan d'Epargne en Actions",
    "CREDIT RENOUVELABLE",
    "CARTE ZEN"
}


contracts = contract.get_contracts()
comptes = compte.get_accounts()
clients = client.get_clients()
tops = top.get_tops()
navigations = navigation.get_navigations()

merged = contracts.merge(comptes, on=['period', 'client_id'], how='left')
merged = merged.merge(clients, on=['period', 'client_id'], how='left')
merged = merged.dropna()
merged = merged.merge(tops, on=['period', 'client_id'], how='left')
merged = merged.merge(navigations, on=['period', 'client_id'], how='left').fillna(False)

merged['contract_type'] = merged['contract_type'].map(lambda x: x if x in Y_filter else '_')
merged.to_csv('merged.csv', index=False, sep=';')
