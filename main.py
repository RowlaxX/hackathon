import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neural_network import MLPClassifier
import joblib
import os
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, mean_squared_error


df = pd.read_csv('merged.csv', sep=';')

def get_xy(product):
    mid = len(df) / 2
    d1 = df[df['contract_type'] == product]

    if len(d1) >= mid:
        m = len(df) - len(d1)
        d1 = d1.sample(n = m)
        d2 = df[df['contract_type'] != product].sample(n = m)
    else:
        d2 = df[df['contract_type'] != product].sample(len(d1))

    d = pd.concat([d1, d2])
    
    X = d.drop(columns=['contract_type', 'period', 'client_id', 'contract_id'], axis=1)
    Y = d['contract_type'].map(lambda x: 1 if x == product else 0)

    return train_test_split(X, Y, test_size=0.2, random_state=42)

def gen_grid():
    return {
            'RandomForestRegressor': {
                'model': RandomForestRegressor(),
                'params': {
                    'n_estimators': [10, 20],
                    'max_depth': [5, 7, None],
                    'min_samples_split': [2, 5]
                }
            },
            'DecisionTreeRegressor': {
                'model': DecisionTreeRegressor(),
                'params': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'MLPRegressor': {
                'model': MLPRegressor(),
                'params': {
                    'hidden_layer_sizes': [(32,64,32), (16,16,16,16)],
                    'activation': ['relu'],
                    'solver': ['adam'],
                    'learning_rate': ['constant', 'adaptive']
                }
    }
    }

def find_best_model_for_product(product):
    # Obtenir les ensembles d'entraînement pour le produit
    X_train, _, Y_train, _ = get_xy(product)

    best_score = None
    best_model = None

    # Boucle sur chaque modèle et ses paramètres
    for model_name, model_info in gen_grid().items():
        print(f"Training {model_name}...")

        # Utiliser 'neg_mean_squared_error' comme métrique de régression
        grid_search = GridSearchCV(
            estimator=model_info['model'],
            param_grid=model_info['params'],
            cv=5,
            scoring='neg_mean_squared_error',  # Métrique choisie
            n_jobs=-1,
            error_score='raise'
        )

        # Ajuster le modèle sur l'ensemble d'entraînement
        grid_search.fit(X_train, Y_train)

        # Stocker le meilleur modèle et ses performances
        best_params = grid_search.best_params_

        # Inverser le score car 'neg_mean_squared_error' retourne une valeur négative
        current_score = -grid_search.best_score_

        if best_score is None or current_score < best_score:  # Moins c'est mieux pour MSE
            best_model = grid_search.best_estimator_
            best_score = current_score

        print(f"Mean Squared Error for {model_name}: {current_score}")

    return best_model

products = list(df['contract_type'].unique())
products.remove('_')
models = dict()
mapping = {
    "Livret A" : "livret_a", 
    "Carte Cleo" : "carte_cleo",
    "Livret Dév. Durable et Solidaire" : "livret_dev_durable_et_solidaire",
    "Carte Premier" : "carte_premier",
    "AMP FORMULE ETUDIANTE" : "amp_formule_etudiante",
    "PRÊT IMMOBILIER" : "pret_immobilier",
    "LCL ETUDIANT" : "lcl_etudiant",
    "ASSURANCE HABITATION" : "assurance_habitation",
    "Prêt Personnel Etudes" : "pret_personnel_etudes",
    "LCL VIE ET PREVOYANCE" : "lcl_vie_et_prevoyance",
    "ASSURANCE DECOUVERT AUTORISE HF" : "assurance_decouvert_autorise_hf",
    "LCL VIE JEUNES" : "lcl_vie_jeunes",
    "ASSURANCE EMPRUNTEUR LCL" : "assurance_emprunteur_lcl",
    "CAPITAL DECES" : "capital_deces",
    "Livret épargne populaire" : "livret_epargne_populaire",
    "Livret Jeune" : "livret_jeune",
    "Prêt Personnel Budget" : "pret_personnel_budget",
    "ASSURANCE REVENUS" : "assurance_revenus",
    "ASSURANCE AUTOMOBILE" : "assurance_automobile",
    "Prêt Personnel Auto" : "pret_personnel_auto",
    "Plan d'épargne logement" : "plan_epargne_logement",
    "Plan d'Epargne en Actions" : "plan_epargne_en_actions",
    "CREDIT RENOUVELABLE" : "credit_renouvelable",
    "CARTE ZEN" : "carte_zen"
}

for product in products:
    print(f'Getting model for product {product}')
    mapped = mapping[product]
    filename = f'models/{mapped}.model'

    if os.path.isfile(filename):
        models[mapped] = joblib.load(filename)
    else:
        model = find_best_model_for_product(product)
        models[mapped] = model
        joblib.dump(model, filename)

def truncate_to_two_decimals(number):
    return int(number * 100) 

app = Flask(__name__)

@app.route('/predict')
def my_endpoint():
    query_param = request.args.get('clientId')
    if query_param:
        return jsonify(predict(query_param))
    else:
        return "No query parameter provided."
    
@app.route('/clients', methods=['GET'])
def get_clients():
    client_ids = list(set(df['client_id'].head(500).to_list()))
    return jsonify(client_ids)

def predict(client_id):
    client = df[df['client_id'] == client_id].drop(columns=['contract_type', 'period', 'client_id', 'contract_id'], axis=1)

    if len(client) == 0:
        raise Exception('No such client found')

    result = dict()
    for name, model in models.items():
        prediction = model.predict(client)
        result[name] = truncate_to_two_decimals(sum(prediction) / len(prediction))

    return result
        

if __name__ == '__main__':
    app.run(debug=True, port=8090)

