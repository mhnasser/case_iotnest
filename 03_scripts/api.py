from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

# Carregar o modelo pré-treinado na seção 3
model = joblib.load('../04_modelos/best_RFC_model.pkl')

# Inicializar o aplicativo Flask
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    """Verifica se o serviço está em execução."""
    return "Serviço em execução", 200

@app.route('/predict', methods=['POST'])
def predict():
    """Recebe entrada JSON e retorna previsões."""
    try:
        # Obter os dados enviados na requisição
        data = request.get_json()

        if not data or 'inputs' not in data:
            return jsonify({'error': 'Entrada JSON inválida. Certifique-se de incluir "inputs".'}), 400

        # Converter os dados em um formato que o modelo aceita
        inputs = np.array(data['inputs'])

        # Fazer previsões
        predictions = model.predict(inputs)

        # Retornar previsões como JSON
        return jsonify({'predictions': predictions.tolist()}), 200

    except Exception as e:
        return jsonify({'error': f'Ocorreu um erro: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(debug=True)
