from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import numpy as np
from joblib import load

app = Flask(__name__)
api = Api(app)

model = load('results/santander_model.pkl')


class Predict(Resource):
    @staticmethod
    def post():
        data = request.get_json(force=True)
        response = {}
        for key, value in data.items():
            id_code = key
            var = [val for idx, val in value.items()]
            prediction = model.predict(np.array(var).reshape(1, -1)).item()
            response.update({id_code: prediction})

        return jsonify(response)


RESOURCES = [(Predict, '/predictions'), ]

for resource, endpoint in RESOURCES:
    api.add_resource(resource, endpoint)

if __name__ == '__main__':
    app.run(debug=True)
