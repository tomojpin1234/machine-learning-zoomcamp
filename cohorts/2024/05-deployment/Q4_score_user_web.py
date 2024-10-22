import pickle
import os
from flask import Flask, jsonify, request


app = Flask("score_user")


def load_dv(name: str):
    with open(name, "rb") as f_in:
        return pickle.load(f_in)
        
def load_model(name: str):
    with open(name, "rb") as f_in:
        return pickle.load(f_in)
    
def get_dv_model():
    dv = load_dv(name="dv.bin")    
    model_name = os.getenv("MODEL_NAME", "model1.bin") 
    print("model_name: ", model_name)
    model = load_model(model_name)
    return dv, model

@app.route("/score", methods=["POST"])
def score_user():
    user = request.get_json()
    dv, model = get_dv_model()
    X = dv.transform([user])
    y_pred = model.predict_proba(X)[0, 1]
    print(y_pred)
    result = {
        "score": y_pred
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
