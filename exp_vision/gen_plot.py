import matplotlib.pyplot as plt
import os
import numpy as np 
import torch

def get_data(path):
    base = np.load(path + "/outputs/base.npy")
    attribution = np.load(path + "/outputs/attribution.npy")
    segmentation = np.load(path + "/outputs/segmentation.npy")
    return torch.from_numpy(base), torch.from_numpy(attribution), torch.from_numpy(segmentation)

def get_top_k_relative(attribution, segmentation, tops):
    n, c, h, w = attribution.size()
    top_k = attribution.reshape(-1, h * w).argsort(dim=-1, descending=True)
    seg_mean = segmentation.reshape(-1, h*w).mean(dim=-1)

    accs = []
    for i in range(n):
        if seg_mean[i] > 0 and seg_mean[i] < 1:
            attr, seg, = attribution[i].reshape(-1), segmentation[i].reshape(-1)
            top_acc = []
            for top in tops:
                n_pixels = int(seg.sum() * top)
                t = top_k[i].reshape(-1)[:n_pixels]
                mask = torch.zeros_like(attr).scatter(dim=-1, index=t, src=torch.ones_like(t).float())
                acc = (seg * mask).sum() / (mask.sum())
                top_acc.append(acc)

            accs.append(top_acc)
    return torch.tensor(accs).mean(dim=0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--path', 
                    nargs='?', default='tmp/')
    parser.add_argument('-n', '--name', 
                    nargs='?', default='output.png')

    args = parser.parse_args()
    top_k = [0.05 + i*0.05 for i in range(18)]
    path = args.path
    folders = sorted([path + "/" + f for f in os.listdir(path)])

    accs = []
    for folder in folders:
        print("\n", folder)
        base, attribution, segmentation = get_data(folder)
        acc = get_top_k_relative(attribution, segmentation, top_k)
        print(acc)
        accs.append(acc)

    try:
        os.mkdir("plot/")
    except: pass 
    top_k = [int(k*100) for k in top_k]
    for folder, acc in zip(folders, accs):
        plt.plot(top_k, acc, label=folder.split("/")[-1].split("_")[0])

    plt.xlabel("Keeping top% pixels relative to mask size")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    
    fig_file = "plot/" + args.name
    print("Saved:", fig_file)
    plt.savefig(fig_file)