import os
import sys

import numpy as np
import pandas as pd

datapath = "/research/cbim/vast/an499/papers/buddy/empatheticdialogues"

#read the data with a max_num to limit number of samples useful for debugging
def get_csv(splitname, max_num = -1):
    prompt_result, response_result, labels_result = [],[], []
    df = open(os.path.join(datapath, f"{splitname}.csv")).readlines()
    for i in range(1, len(df)):
        if max_num != -1 and i>=max_num:
            break
        data_line = df[i].strip().split(",")
        context = data_line[2]
        prompt = data_line[3].replace("_comma_", ",")
        response = data_line[5].replace("_comma_", ",")
        prompt_result.append(prompt)
        response_result.append(response)
        labels_result.append(context)
    return prompt_result, response_result, labels_result

def get_file(max_num_tr=-1, max_num_val=-1):
    print("Reading data started finished ...")
    prompt_train, response_train, labels_train = get_csv("train",max_num = max_num_tr)
    prompt_val, response_val, labels_val = get_csv("valid",max_num = max_num_val)
    print("Reading data finished ...")
    return prompt_train, response_train, labels_train, prompt_val, response_val, labels_val
#get_file()