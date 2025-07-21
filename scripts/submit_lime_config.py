import pickle
import requests


key_names = [
    'West_Highland_white_terrier',
    'American_coot', 
    'racer',  
    'flamingo',  
    'kite',  
    'goldfish',
    'tiger_shark',
    'vulture',  
    'common_iguana',  
    'orange'
]

params = {
    "labels": None,  
    "top_labels": 1,  
    "hide_color": 0,  
    "num_features": 100000,  
    "num_samples": 1000,  
    "batch_size": 10,  
    "segmentation_fn": None, 
    "distance_metric": "cosine",  
    "model_regressor": None, 
    "random_seed": None,  
}

all_params = {}
for i in key_names:
    all_params[i] = params

with open("./content/explain_params.pkl", "wb") as f:
    pickle.dump(all_params, f)

response = requests.post("http://34.122.51.94:9091/lime", files={"file": open("./content/explain_params.pkl", "rb")}, headers={"token": "17805920"})
print(response.json())

# Result of submission : {'avg_iou': 0.3081150221402621, 'avg_time': 4.651949858665466}
