import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

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
        'RandomForestClassifier': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [10, 20],
                'max_depth': [5, 7, None],
                'min_samples_split': [2, 5]
            }
        },
        'DecisionTreeClassifier': {
            'model': DecisionTreeClassifier(),
            'params': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'MLPClassifier': {
            'model': MLPClassifier(),
            'params': {
                'hidden_layer_sizes': [(32,64,32), (64,32), (32,32,32),(16,16,16,16) ],
                'activation': ['tanh', 'relu'],
                'solver': ['adam'],
                'learning_rate': ['constant', 'adaptive']
                #'early_stopping': [True],  # Enable early stopping
                #'n_iter_no_change': [5, 10]  # Number of iterations with no improvement to trigger stopping
            }
        }
    }

def find_best_model_for_product(product):
    X_train, _, Y_train, _ = get_xy(product)

    best_score = None
    best_model = None

    for model_name, model_info in gen_grid().items():
        print(f"Training {model_name}...")

        grid_search = GridSearchCV(
            estimator=model_info['model'],
            param_grid=model_info['params'],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            error_score='raise'
        )

        grid_search.fit(X_train, Y_train)

        # Stocker le meilleur modèle et ses performances
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        if best_score is None or grid_search.best_score_ > best_score:
            best_model = grid_search.best_estimator_
            best_score = grid_search.best_score_

        print(f"Score for {model_name}: {grid_search.best_score_}")
    
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

import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "lcl-hackathon-e7-sbox-a647-6d503291ee1a.json"

PROJECT_ID = "lcl-hackathon-e7-sbox-a647"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

model = GenerativeModel("gemini-1.5-pro-001", generation_config=GenerationConfig(temperature=0.4))
chat = model.start_chat()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/predict')
def my_endpoint():
    query_param = request.args.get('clientId')
    if query_param:
        predictions = predict(query_param)
        if predictions:
            ai_response = get_ai_response(predictions)
            return jsonify({"predictions": predictions, "ai_response": ai_response})
        else:
            return "No such client found.", 404
    else:
        return "No query parameter provided.", 400

def predict(client_id):
    client = df[df['client_id'] == client_id].drop(columns=['contract_type', 'period', 'client_id', 'contract_id'], axis=1)

    if len(client) == 0:
        raise Exception('No such client found')
    
    result = dict()
    for name, model in models.items():
        prediction = model.predict(client)
        result[name] = bool((sum(prediction) / len(prediction)) > 0.5)

    return result

@app.route('/clients', methods=['GET'])
def get_clients():
    client_ids = df['client_id'].head(200).tolist()
    return jsonify(client_ids)

def get_ai_response(predictions):
    prompt = f"Comporte-toi comme un conseiller bancaire qui propose des produits à un client. Les produits sont : {predictions}."
    response = chat.send_message(prompt)
    return response.candidates[0].content.parts[0]


if __name__ == '__main__':
    app.run(debug=True, port=8080)

