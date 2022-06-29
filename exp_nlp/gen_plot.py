
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
import json
import os
import matplotlib.pyplot as plt
from attribution import *
import numpy as np

def get_data(path):

    with open(path, "r", encoding="utf-8") as f:
        file = json.load(f)

    labels = file["labels"]
    scores = file["scores"]
    return scores, labels


tokenizer = RobertaTokenizerFast.from_pretrained("textattack/roberta-base-imdb")
model = RobertaForSequenceClassification.from_pretrained("textattack/roberta-base-imdb")



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--path', 
                    nargs='?', default='tmp/')
    parser.add_argument('-n', '--name', 
                    nargs='?', default='output.png')

    args = parser.parse_args()
    path = args.path
    files = sorted([path + "/" + f for f in os.listdir(path)])


    file_name = "plot/" + args.name
    fig, axs = plt.subplots(1, 2, figsize=(16, 4))
    try:
        os.makedirs("plot/")
    except:
        pass

    top_k = [0.05 + i*0.05 for i in range(9)]
    m = EvalAttributionModelWithTransformers(model, tokenizer)

    all_accs = []
    for i, file in enumerate(files):
        print(file)
        scores, labels = get_data(file)

        accs = []
        for k in top_k:
            acc = m.fit(scores, labels, top=k, descending=True)
            accs.append(acc)

        all_accs.append(accs)
        axs[0].plot(top_k, all_accs[i], label=file.split("/")[-1].split("_")[0])
    

    top_k = [0.05 + i*0.05 for i in range(15)]
    m = EvalAttributionModelWithTransformers(model, tokenizer, reversed=True)

    all_accs_reversed = []
    for i, file in enumerate(files):
        print(file)
        scores, labels = get_data(file)

        accs_reversed = []
        for k in top_k:
            acc = m.fit(scores, labels, top=k, descending=True)
            accs_reversed.append(acc)

        all_accs_reversed.append(accs_reversed)
        axs[1].plot(top_k, all_accs_reversed[i], label=file.split("/")[-1].split("_")[0])

    
    axs[1].legend(loc="best")
    plt.tight_layout()
    fig.savefig(file_name)

