import firebase_admin
from firebase_admin import credentials, db
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from joblib import dump
from flask import Flask,request,jsonify
from joblib import load
import pickle
import os

def train_model():
    #pull database data
    ref = db.reference('/')
    data = ref.get()

    #data filtering and swipe counting
    cleaned_data = {}
    
    required_keys = ["averageDirection","averageVelocity","directionEndToEnd",
                    "midStrokeArea", "midStrokePressure","pairwiseVelocityPercentile",
                    "startX","startY","stopX","stopY","strokeDuration"]
    #iterate over users
    for user, user_swipes in data.items():
        valid_swipes = []

        for swipe_id, swipe_data in user_swipes.items():
            if all(key in swipe_data for key in required_keys):
                valid_swipes.append(
                    [swipe_data[key] for key in required_keys]
                )

        if len(valid_swipes) >= 50:
            cleaned_data[user] = np.array(valid_swipes[:50])


    #create training data and labels
    users = set(cleaned_data.keys())
    #save users for endpoint
    with open("user.pkl", "wb") as f:
        pickle.dump(users, f)

    x = []
    y = []

    for user in users:
        swipes = cleaned_data[user]

        for swipe in swipes:
            x.append(swipe)     
            y.append(int(user)) 

    #begin training              
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,random_state=42)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    mlp = MLPClassifier(
        hidden_layer_sizes= (16, 8),  # two hidden layers: 32 + 16 neurons
        activation='relu',            # ReLU activation
        solver='adam',                # Adam optimizer
        max_iter = 150,               # max epochs
        random_state=42,
        alpha = 0.001,
        early_stopping = True
    )

    mlp.fit(x_train, y_train)

    #make sure model is accurate
    y_pred = mlp.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    #save the model
    dump(mlp, "mlp.joblib")
    #save the scaler
    dump(scaler, "scaler.joblib")


# endpoint creation

app = Flask(__name__)

#initially load mlp model and scaler if it exists
if os.path.exists("mlp.joblib"):
    mlp = load("mlp.joblib")
    scaler = load("scaler.joblib")

#load firebase once
cred = credentials.Certificate("firebase_service_key.json")
firebase_admin.initialize_app(cred, {'databaseURL':
    'your firebase URL here'})

#open user file if previously there
if os.path.exists("user.pkl"):
    with open("user.pkl", "rb") as f:
        user = pickle.load(f)
else:
    user = set()

# I would never set up an endpoint in this fashion normally, but its only pinged after enrollment so there is not a good workaround to train the model
# before a user is at 50 swipes.
@app.route('/authenticate/<userID>', methods=["POST"])
def authenticate(userID):
    global mlp, scaler, user

    data = request.get_json()
    required_keys = ["averageDirection","averageVelocity","directionEndToEnd",
                     "midStrokeArea", "midStrokePressure","pairwiseVelocityPercentile",
                     "startX","startY","stopX","stopY","strokeDuration"]

    if not all(k in data for k in required_keys):
        return jsonify({"message": "Invalid features provided"}), 400

    # retrain if new user
    if userID not in user:
        train_model()
        mlp = load("mlp.joblib")
        scaler = load("scaler.joblib")
        with open("user.pkl", "rb") as f:
            user = pickle.load(f)

    x = np.array([data[k] for k in required_keys]).reshape(1, -1)

    x = scaler.transform(x)

    pred_user = mlp.predict(x)[0]
    confidence = mlp.predict_proba(x).max()

    #confidence value chosen arbitrarily, can be adjusted
    match = (pred_user == int(userID) and confidence >= .80)
    
    return jsonify({
        "match": bool(match),
        "message": "Matched" if match else "Not matched"
    }), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)








        

        

  

