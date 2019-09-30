import pandas as pd
import requests
import json

url = 'http://127.0.0.1:5000/predictions'

test = pd.read_csv('data/X_test.csv')
test.set_index("ID_code", inplace=True)  # el id cómo índice

j_data = json.loads(test.to_json(orient='index'))
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
req = requests.post(url, json=j_data, headers=headers)
print(req, req.text)
