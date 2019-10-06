import pandas as pd
import requests
import json
import time
start_time = time.time()

# Local
# url = 'http://127.0.0.1:5000/predictions'
# Server

url = 'https://santander-api-faa.herokuapp.com/predictions'

test = pd.read_csv('data/X_test.csv', nrows=100)
test.set_index("ID_code", inplace=True)  # el id cómo índice

j_data = json.loads(test.to_json(orient='index'))
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
start_time = time.time()
req = requests.post(url, json=j_data, headers=headers)
print(req, req.text)
print("--- %s segundos ---" % (time.time() - start_time))
