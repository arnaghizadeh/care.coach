import json

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch


# load the model once
from FinalSubmission.Part3_main import BaseAgent
from FinalSubmission.custom_dataset import tokenizer
from FinalSubmission.gpt2_model import model


model.load_state_dict(torch.load("gpt2_model.pth"))
model.eval()


ba = BaseAgent(tokenizer, model,device = 2)

def run_test(text):
    predict_B = ba.get_response("A: "+ text + "B: ", max_len=52)
    predict_lbl = predict_B.split("B: ")
    if len(predict_lbl)>1:
        predict_lbl = predict_lbl[1]
        predict_lbl = predict_lbl.split("<EMOTION_TYPE> ")
        if len(predict_lbl)>1:
            predict_lbl = predict_lbl[0]
    predict_lbl = "".join(predict_lbl)
    result = predict_lbl
    return result


app = Flask(__name__)
cors = CORS(app)


# Create the receiver API POST endpoint:
@app.route("/receiver", methods=["POST"])
def postME():
    # get the data from client, run it through NN classifier, return the result in a json file to client
    data = request.get_json()
    data = run_test(json.dumps(data))
    data = jsonify(data)
    return data


if __name__ == "__main__":
    app.run(debug=True)
