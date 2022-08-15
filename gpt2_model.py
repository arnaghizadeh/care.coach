import os
from transformers import GPT2LMHeadModel

modelpath = "/research/cbim/vast/an499/papers/buddy/modelstates/hugface_models"
def get_model():
    pretrained_hf_model = 'gpt2'
    cache_dir = os.path.join(modelpath,pretrained_hf_model,)
    model = GPT2LMHeadModel.from_pretrained(pretrained_hf_model, cache_dir=cache_dir,)
    return model
model = get_model()