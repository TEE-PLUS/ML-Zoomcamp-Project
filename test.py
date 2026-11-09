import requests

url = 'http://localhost:9696/predict'

customer = {
    'highbp': 1.0,
    'highchol': 1.0,
    'cholcheck': 1.0,
    'bmi': 25.0,
    'smoker': 0.0,
    'stroke': 0.0,
    'heartdiseaseorattack': 1.0,
    'physactivity': 1.0,
    'fruits': 1.0,
    'veggies': 0.0,
    'hvyalcoholconsump': 0.0,
    'anyhealthcare': 1.0,
    'nodocbccost': 0.0,
    'genhlth': 2.0,
    'menthlth': 0.0,
    'physhlth': 0.0,
    'diffwalk': 0.0,
    'sex': 0.0,
    'age': 9.0,
    'education': 6.0,
    'income': 8.0}

response = requests.post(url, json=customer).json()
print(response)

if response['predictions'] == 1:
    print('Have prediabetes or diabetes')
else:
    print('No prediabetes or diabetes')