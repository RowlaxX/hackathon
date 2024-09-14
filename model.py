import pandas as pd
import numpy as np
from keras import layers, losses, optimizers, Sequential
from sklearn.model_selection import train_test_split

NB_INPUT = 10
NB_PRODUCT = 32

_model = Sequential(layers=[
    layers.Dense(64, activation='relu', input_shape=(NB_INPUT, )),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(NB_PRODUCT, activation='softmax')
])

_model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001), 
    loss=losses.mean_squared_error,
    metrics=['accuracy']
)

def train(df: pd.DataFrame):
    X = df.drop()
    X_train, X_test, y_train, y_test = train_test_split()
    _model.fit

def predict(df: pd.DataFrame):
    pass

df1 = pd.read_csv('Synth_Contrat.csv', sep=';')
df2 = pd.read_csv('client_clean.csv', sep=';')

df1['period'] = df1['period'].astype(str)
df2['period'] = df2['period'].astype(str)

pd.merge(df1, df2, how='left', on=['period', 'id']).to_csv('test.csv', index=False)