import sklearn
from flask import Flask, request, jsonify
import pandas as pd
import joblib


app = Flask(__name__)

model = joblib.load('C:\\Users\\tearo\\PycharmProjects\\ModuloFIA\\ModuloFIA\\modelloML\\random_forest_model.joblib')
label_encoder = joblib.load('C:\\Users\\tearo\\PycharmProjects\\ModuloFIA\\ModuloFIA\\modelloML\\label_encoder.joblib')

@app.route('/predictCategoria', methods=['POST'])
def predict():

    colonne_vuote = ['Gender', 'Married', 'Graduated', 'Profession', 'Budget', 'Age', 'Family_Size']
    dataset_colonne = pd.DataFrame(columns=colonne_vuote)

    dati_utente = request.json

    user_input_df = pd.DataFrame([dati_utente])

    for col in dataset_colonne.columns:
        if col in user_input_df.columns:
            dataset_colonne[col] = user_input_df[col]
        else:
            dataset_colonne[col] = 0

    input_df_template = dataset_colonne.reindex(columns=colonne_vuote).fillna(0)

    categoria_predetta_codificata = model.predict(input_df_template)

    categoria_predetta = label_encoder.inverse_transform(categoria_predetta_codificata)[0]

    return jsonify({'categoria_predetta': categoria_predetta})


if __name__ == '__main__':
    app.run(debug=True, port=2000)