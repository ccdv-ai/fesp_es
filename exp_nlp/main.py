
from datasets import load_dataset
import re
from torch.utils.data import DataLoader
from random import shuffle, seed
import os
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import *
from trainer import *
from utils import *
from attribution import *

def clean(text):
    text = text.replace("\n", " ")
    text = re.sub(" +", " ", text)
    text = text.replace("’", "'")
    text = re.sub('<[^>]*>', '', text)
    return text.strip()
    
def f(x):
    return torch.tensor(x).max()

class WrapperModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, inputs_embeds=None):
        return torch.softmax(self.model(inputs_embeds=inputs_embeds)["logits"], dim=-1)

def custom_forward(inputs_embeds=None):
    return torch.softmax(model(inputs_embeds=inputs_embeds)["logits"], dim=-1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", help="path to the trainer config", type=str)
    parser.add_argument("--model_config", help="path to model config", type=str)

    args = parser.parse_args()

    train_config = get_config(args.train_config)
    model_config = get_config(args.model_config)

    # Train config
    set_all_seeds(train_config.seed)
    test_size = train_config.test_size
    batch_size = train_config.batch_size
    max_len = train_config.max_len

    dataset = load_dataset("imdb")
    data = dataset["test"]
    test_data = [(clean(d), l) for d, l in zip(data["text"], data["label"])]
    shuffle(test_data)
    test_data = test_data[:test_size]


    # Model config
    tokenizer = RobertaTokenizerFast.from_pretrained("textattack/roberta-base-imdb")
    model = RobertaForSequenceClassification.from_pretrained("textattack/roberta-base-imdb")

    
    tokenizer.model_max_length = max_len
    tokenizer.init_kwargs['model_max_length'] = max_len

    test_dataset = DataLoader(test_data, batch_size=1)
    test_dataset_attr = DataLoader(test_data, batch_size=test_size)

    s, labels = next(iter(test_dataset_attr))
    s = list(s)

    # Params
    whole_word = model_config.whole_word
    try:
        kernel, stride = model_config.kernel, model_config.stride
        print(kernel, stride)
    except:
        pass 
    model_name = model_config.model


    try:
        os.makedirs("tmp/")
    except:
        pass
    
    if model_name == "fesp":
        attr_model = FESPWithTransformers(
            model=model, 
            tokenizer=tokenizer, 
            batch_size=batch_size
            )
        outputs = attr_model.fit(s, kernel, stride, whole_word)
        final_outputs = attr_model.merge_scores(outputs, f, whole_word)
        attr_model.save_to_file(f"tmp/fesp_{max_len}_{kernel}_{stride}_{whole_word}.json", final_outputs, labels, "fesp")
        del attr_model

    elif model_name == "equalsurplus":
        attr_model = ESWithTransformers(
            model=model, 
            tokenizer=tokenizer, 
            batch_size=batch_size
            )
        outputs = attr_model.fit(s, kernel, stride, whole_word)
        final_outputs = attr_model.merge_scores(outputs, f, whole_word)
        attr_model.save_to_file(f"tmp/es_{max_len}_{kernel}_{stride}_{whole_word}.json", final_outputs, labels, "es")
        del attr_model

    elif model_name == "shapexplainer":
        attr_model = ShapExplainerWrapperWithTransformers(
            model=model, 
            tokenizer=tokenizer
            )
        outputs = attr_model.fit(s)
        final_outputs = attr_model.merge_scores(outputs, f, whole_word)
        attr_model.save_to_file(f"tmp/shapexplainer_{max_len}.json", final_outputs, labels, "shapexplainer")
        del attr_model

    elif model_name in ["deeplift", "gradientshap", "occlusion", "integratedgradient"]:

        wrapped_model = WrapperModel(model)


        if model_name == "integratedgradient":
            from captum.attr import IntegratedGradients

            ig = IntegratedGradients(wrapped_model)
            attr_model = CaptumWrapperWithTransformers(
                model=wrapped_model, 
                captum_model=ig, 
                tokenizer=tokenizer, 
                embedding_layer=model.roberta.embeddings
                )
            outputs = attr_model.fit(s)
            final_outputs = attr_model.merge_scores(outputs, f, whole_word)
            attr_model.save_to_file(f"tmp/ig_{max_len}.json", final_outputs, labels, "integrated_gradient")
            del attr_model

        elif model_name == "gradientshap":
            from captum.attr import GradientShap

            ig = GradientShap(wrapped_model)
            attr_model = CaptumWrapperWithTransformers(
                model=wrapped_model, 
                captum_model=ig, 
                tokenizer=tokenizer, 
                embedding_layer=model.roberta.embeddings
                )
            outputs = attr_model.fit(s)
            final_outputs = attr_model.merge_scores(outputs, f, whole_word)
            attr_model.save_to_file(f"tmp/gradientshap_{max_len}.json", final_outputs, labels, "gradientshap")
            del attr_model

        elif model_name == "deeplift":
            from captum.attr import DeepLift

            ig = DeepLift(wrapped_model)
            attr_model = CaptumWrapperWithTransformers(
                model=wrapped_model, 
                captum_model=ig, 
                tokenizer=tokenizer, 
                embedding_layer=model.roberta.embeddings
                )
            outputs = attr_model.fit(s)
            final_outputs = attr_model.merge_scores(outputs, f, whole_word)
            attr_model.save_to_file(f"tmp/deeplift_{max_len}.json", final_outputs, labels, "deeplift")
            del attr_model

        elif model_name == "occlusion":
            from captum.attr import Occlusion

            ig = Occlusion(wrapped_model)
            attr_model = CaptumWrapperWithTransformers(
                model=wrapped_model, 
                captum_model=ig, 
                tokenizer=tokenizer, 
                embedding_layer=model.roberta.embeddings
                )
            outputs = attr_model.fit(s, kernel=kernel, stride=stride)
            final_outputs = attr_model.merge_scores(outputs, f, whole_word)
            attr_model.save_to_file(f"tmp/occlusion_{max_len}_{kernel}_{stride}.json", final_outputs, labels, "occlusion")
            del attr_model
    

    





