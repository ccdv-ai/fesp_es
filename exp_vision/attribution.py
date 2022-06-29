import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from math import sqrt
from torch.distributions.categorical import Categorical
from random import randrange
import logging 

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    Occlusion
)

import shap



class BaseExplanationModel():

    def __init__(self, model, criterion=None, noise_distribution=(0, 0), batch_size=64, device="cuda"):
        
        self.model = model.to(device)
        self.noise_distribution = noise_distribution
        self.batch_size = batch_size
        self.device = device

        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = nn.CrossEntropyLoss(reduction="none")

    def set_distribution(self, sample):
        n, c, h, w = sample.size()
        self.noise = torch.normal(self.noise_distribution[0], self.noise_distribution[1], size=(n, c, h, w), device=self.device)
    
    def initial_step(self, sample, return_grad=False):

        if return_grad:
            sample = torch.autograd.Variable(sample.to(self.device), requires_grad=True)

        outputs = self.model(sample.to(self.device))
        label = outputs.argmax(dim=-1)
        loss = self.criterion(outputs, label).flatten()

        if return_grad:
            loss.backward()
            return loss.detach(), label.detach(), sample.grad.data.clone().detach()
        return loss, label

    def step(self, sample, label):
        
        outputs = self.model(sample)
        loss = self.criterion(outputs, label)
        loss = loss.reshape(-1, 1, 1, 1).detach()
        return loss

    def get_noise(self, mask):
        noise = torch.normal(torch.zeros_like(mask) + self.noise_distribution[0], std=torch.zeros_like(mask) + self.noise_distribution[1])
        noise_mask = noise.clone()
        noise_mask[noise_mask != 0] = 1
        return noise, noise_mask

    def get_custom_block_mask(self, image_shape=(3, 32, 32), block_size=(1, 1), stride=1, full_mask_only=False, circular=False, smoothing_degree=None):

        c, h, w = image_shape 
        n_h_chunks = h // stride 
        n_w_chunks = w // stride

        mask = torch.zeros(h, w)
        mask[:block_size[0], :block_size[1]] = 1

        outputs = []
        
        if full_mask_only:
            for i in range(n_h_chunks - block_size[0]//stride + 1):
                if i > 0:
                    mask = torch.roll(mask, dims=-2, shifts=stride)

                for j in range(n_w_chunks - block_size[1]//stride + 1):
                    if j > 0:
                        mask = torch.roll(mask, dims=-1, shifts=stride)

                    outputs.append(mask)

                mask = torch.roll(mask, dims=-1, shifts=block_size[0])
        
        elif circular:
            for i in range(n_h_chunks):
                if i > 0:
                    mask = torch.roll(mask, dims=-2, shifts=stride)

                for j in range(n_w_chunks):
                    if i != 0 or j != 0 :
                        mask = torch.roll(mask, dims=-1, shifts=stride)

                    outputs.append(mask)
        
        else:
            mask = torch.zeros(h, w)
            for i in range(n_h_chunks + block_size[0]//stride - 1):
                for j in range(n_w_chunks + block_size[1]//stride - 1):
                    new_mask = mask.clone()
                    i_0 = max(-block_size[0] + stride*(i + 1), 0)
                    i_1 = min(stride*(i + 1), h)

                    j_0 = max(-block_size[1] + stride*(j + 1), 0)
                    j_1 = min(stride*(j + 1), h)
                    new_mask[i_0:i_1, j_0:j_1] = 1
                    outputs.append(new_mask)
        
        mask = torch.stack(outputs).unsqueeze(1)

        if smoothing_degree is not None and smoothing_degree > 0: 
            print("is smoothed")
            h, w = block_size[0], block_size[1]
            if h % 2 != 1:
                h -= 1 
            if w % 2 != 1:
                w -= 1 
            pooling = nn.AvgPool2d((h, w), 1, padding=(h//2, w//2)).to(self.device)
            mask = self.process_pooling(mask, pooling)**smoothing_degree

        return mask

    def process_pooling(self, data, pooling):
        data = data.split(16, dim=0)
        outputs = []
        for d in data:
            output = pooling(d.to(self.device))
            outputs.append(output.cpu())
        return torch.cat(outputs, dim=0)

    def set_base_mask(self, c, h, w):
        mask = torch.zeros(h*w, 1, h*w)
        idx = torch.arange(h*w).unsqueeze(-1).unsqueeze(-1)
        self.base_mask = torch.scatter(mask, dim=-1, index=idx, src=torch.ones_like(idx).float()).reshape((h*w, 1, h, w))

        print("base_mask_shape", self.base_mask.shape)

    def extract_features(self, data, top=0., use_segmentation=False, batch_size=32):

        imgs, masks, labels = [], [], []
        for d in data:
            _, img, __, shapley, label = d

            _, c, h, w = shapley.size()

            topk = int(__.mean()*w*h) if use_segmentation else int(top*w*h)
            shapley = shapley.reshape(-1)
            idx = shapley.argsort(dim=-1, descending=True)[:topk]
            #idx = shapley.argsort(dim=-1, descending=False)[:topk]

            mask = torch.zeros_like(shapley).scatter(dim=-1, index=idx, src=torch.ones_like(idx).float())
            masks.append(mask.reshape(1, 1, h, w))

            imgs.append(img)
            labels.append(label)

        return DataLoader(
            TensorDataset(torch.cat(imgs, dim=0), torch.cat(labels, dim=0), torch.cat(masks, dim=0)), 
            batch_size=batch_size
            )
    
    def segmentation_accuracy(self, data):
        
        accs = 0
        for d in data:
            _, img, __, shapley, label = d

            _, c, h, w = shapley.size()

            topk = int(__.mean()*w*h) 
            shapley = shapley.reshape(-1)
            idx = shapley.argsort(dim=-1, descending=True)[:topk]

            mask = torch.zeros_like(shapley).scatter(dim=-1, index=idx, src=torch.ones_like(idx).float())
            accs += (mask == __.reshape(-1)).float().mean()

        return float(accs) / len(data)


class IntegratedGradientsModel(BaseExplanationModel):

    def __init__(self, model, criterion=None, noise_distribution=None, device="cuda"):

        super().__init__(model, criterion, noise_distribution, device)

    def fit(self, data_loader, n_samples=16, zero_baseline=True):
        
        self.model.eval()
        outputs = []

        for i, data in enumerate(data_loader):
            _, img, __, ___ = data 

            shapley_score, label = self._fit(_, img, zero_baseline)
            outputs.append((_, img.cpu(), __, shapley_score.detach().cpu(), label.detach().cpu()))

            if i + 1 == n_samples:
                break
        return outputs

    def _fit(self, raw_img, sample, zero_baseline):
        
        with torch.no_grad():
            self.set_distribution(sample)

            ig = IntegratedGradients(self.model)
            sample = sample.to(self.device)
            label = self.model(sample).argmax(dim=-1)

            if zero_baseline:
                attributions = ig.attribute(sample, baselines=torch.zeros_like(sample), target=label, return_convergence_delta=False).float()
            else:
                attributions = ig.attribute(sample, sample, target=label, return_convergence_delta=False).float()

        return attributions.mean(dim=1, keepdim=True), label


class GradientShapModel(BaseExplanationModel):

    def __init__(self, model, criterion=None, noise_distribution=None, device="cuda"):

        super().__init__(model, criterion, noise_distribution, device)

    def fit(self, data_loader, n_samples=16, n_reestimations=16, zero_baseline=True):
        
        self.model.eval()
        outputs = []

        for i, data in enumerate(data_loader):
            _, img, __, ___ = data 

            shapley_score, label = self._fit(_, img, n_reestimations, zero_baseline)
            outputs.append((_, img.cpu(), __, shapley_score.detach().cpu(), label.detach().cpu()))

            if i + 1 == n_samples:
                break
        return outputs

    def _fit(self, raw_img, sample, n_reestimations, zero_baseline):
        
        with torch.no_grad():
            ig = GradientShap(self.model)
            sample = sample.to(self.device)
            label = self.model(sample).argmax(dim=-1)

            if zero_baseline:
                attributions = ig.attribute(sample, baselines=torch.zeros_like(sample), stdevs=1.0, n_samples=n_reestimations, target=label, return_convergence_delta=False).float()
            else:
                attributions = ig.attribute(sample, baselines=sample, stdevs=1.0, target=label, n_samples=n_reestimations, return_convergence_delta=False).float()
        return attributions.mean(dim=1, keepdim=True), label


class DeepLiftModel(BaseExplanationModel):

    def __init__(self, model, criterion=None, noise_distribution=None, device="cuda"):

        super().__init__(model, criterion, noise_distribution, device)

    def fit(self, data_loader, n_samples=16, zero_baseline=True):
        
        self.model.eval()
        outputs = []

        for i, data in enumerate(data_loader):
            _, img, __, ___ = data 

            shapley_score, label = self._fit(_, img, zero_baseline)
            outputs.append((_, img.cpu(), __, shapley_score.detach().cpu(), label.detach().cpu()))

            if i + 1 == n_samples:
                break
        return outputs

    def _fit(self, raw_img, sample, zero_baseline):
        
        with torch.no_grad():
            ig = DeepLift(self.model)
            sample = sample.to(self.device)
            label = self.model(sample).argmax(dim=-1)

            if zero_baseline:
                attributions = ig.attribute(sample, baselines=torch.zeros_like(sample), target=label, return_convergence_delta=False).float()
            else:
                attributions = ig.attribute(sample, sample, target=label, return_convergence_delta=False).float()

        return attributions.mean(dim=1, keepdim=True), label


class DeepLiftShapModel(BaseExplanationModel):

    def __init__(self, model, criterion=None, noise_distribution=None, device="cuda"):

        super().__init__(model, criterion, noise_distribution, device)

    def get_distribution(self, data_loader, size=16):
        d = []
        for i, data in enumerate(data_loader):
            _, img, __, ___ = data 
            d.append(img)
            if i == size - 1:
                break
        return torch.cat(d, dim=0)

    def fit(self, data_loader, n_samples=16, zero_baseline=True, baseline_size=16):
        
        self.model.eval()

        d = self.get_distribution(data_loader, baseline_size)

        outputs = []
        for i, data in enumerate(data_loader):
            _, img, __, ___ = data 

            shapley_score, label = self._fit(d, img, zero_baseline)
            outputs.append((_, img.cpu(), __, shapley_score.detach().cpu(), label.detach().cpu()))

            if i + 1 == n_samples:
                break
        return outputs

    def _fit(self, d, sample, zero_baseline):
        
        with torch.no_grad():
            ig = DeepLiftShap(self.model)
            sample = sample.to(self.device)
            label = self.model(sample).argmax(dim=-1)

            if zero_baseline:
                attributions = ig.attribute(sample, baselines=torch.zeros_like(sample.expand(2, -1, -1, -1)), target=label, return_convergence_delta=False).float()
            else:
                attributions = ig.attribute(sample, baselines=d.to(self.device), target=label, return_convergence_delta=False).float()
        return attributions.mean(dim=1, keepdim=True), label


class OcclusionModel(BaseExplanationModel):

    def __init__(self, model, criterion=None, noise_distribution=None, device="cuda"):

        super().__init__(model, criterion, noise_distribution, device)

    def fit(self, data_loader, n_samples=16, strides=8, sliding_window_shapes=(3, 16, 16)):
        
        self.model.eval()
        outputs = []

        for i, data in enumerate(data_loader):
            _, img, __, ___ = data 

            shapley_score, label = self._fit(_, img, strides, sliding_window_shapes)
            outputs.append((_, img.cpu(), __, shapley_score.detach().cpu(), label.detach().cpu()))

            if i + 1 == n_samples:
                break
        return outputs

    def _fit(self, raw_img, sample, strides, sliding_window_shapes):
        
        with torch.no_grad():
            ig = Occlusion(self.model)
            sample = sample.to(self.device)
            label = self.model(sample).argmax(dim=-1)

            attributions = ig.attribute(
                sample, 
                strides=strides,
                sliding_window_shapes=sliding_window_shapes,
                baselines=torch.normal(torch.zeros_like(sample) + self.noise_distribution[0], std=torch.zeros_like(sample) + self.noise_distribution[1]), 
                target=label, 
                ).float()

        return attributions.mean(dim=1, keepdim=True), label


class EqualSurplusModel(BaseExplanationModel):

    def __init__(self, model, criterion=None, noise_distribution=(0, 0), batch_size=64, device="cuda"):

        super().__init__(model, criterion, noise_distribution, batch_size, device)

    def fit(self, data_loader, mask, n_samples=16):
        
        self.model.eval()
        outputs = []

        for i, data in enumerate(data_loader):
            _, img, __, ___ = data 

            shapley_score, label = self._fit(img, mask)

            outputs.append((_, img.cpu(), __, shapley_score.detach().cpu(), label.detach().cpu()))

            if i + 1 == n_samples:
                break
        return outputs

    def _fit(self, sample, mask):

        n = mask.shape[0]

        v_n, predicted_label = self.initial_step(sample)

        with torch.no_grad():
            self.set_distribution(sample)
            loader = DataLoader(
                TensorDataset(sample.expand(mask.shape[0], -1, -1, -1), predicted_label.expand(n).cpu(), mask), 
                batch_size=self.batch_size
                ) 

            with torch.no_grad():
                inf_values = self.shapley_loop(loader)

            h, w = inf_values.shape[-2:]
            #scores = inf_values + (v_n - inf_values.sum()) / inf_values.shape[0]
            scores = inf_values + (v_n - inf_values.sum()) / (h*w)

            return (scores, predicted_label)

    def shapley_loop(self, data):

        masks = 1e-6
        shapley_values = 0

        for batch in data:
            inputs, labels, mask = batch
            inputs, labels, mask = inputs.to(self.device), labels.to(self.device), mask.to(self.device)
            noise = self.noise

            inputs = inputs * mask + (1 - mask) * noise 
            
            predictions = self.model(inputs)
            values = self.criterion(predictions, labels).reshape(-1, 1, 1, 1) * mask

            shapley_values += values.sum(dim=0, keepdim=True)
            masks += mask.sum(dim=0, keepdim=True)

        shapley_values /= masks
        return shapley_values.sum(dim=0, keepdim=True).reshape(1, *shapley_values.size()[-3:])


class FESPModel(BaseExplanationModel):

    def __init__(self, model, criterion=None, noise_distribution=(0, 0), batch_size=64,  device="cuda"):

        super().__init__(model, criterion, noise_distribution, batch_size, device)

    def fit(self, data_loader, mask, n_samples=16):
        
        self.model.eval()
        outputs = []

        for i, data in enumerate(data_loader):
            _, img, __, ___ = data 

            shapley_score, label = self._fit(img, mask)
            outputs.append((_, img.cpu(), __, shapley_score.detach().cpu(), label.detach().cpu()))

            if i + 1 == n_samples:
                break
        return outputs

    def _fit(self, sample, mask):

        n = mask.shape[0]
        v_n, predicted_label = self.initial_step(sample, return_grad=False)

        with torch.no_grad():
            self.set_distribution(sample)
            loader = DataLoader(
                TensorDataset(sample.expand(mask.shape[0], -1, -1, -1), predicted_label.expand(n).cpu(), mask), 
                batch_size=self.batch_size
                ) 

            
            inf_values, sup_values = self.shapley_loop(loader)

            h, w = sample.shape[-2:]
            sup_values = (0 - sup_values)

            inf_sum = inf_values.sum()
            sup_sum = sup_values.sum()
            
            w = (v_n - sup_sum) / (inf_sum - sup_sum)
            scores = w * inf_values + (1 - w) * sup_values

            return (scores, predicted_label)

    def shapley_loop(self, data):

        inputs, labels, mask = next(iter(data))

        masks = 1e-6
        masks2 = 1e-6
        shapley_values_inf = 0
        shapley_values_sup = 0

        for batch in data:
            inputs, labels, mask = batch
            inputs, labels, mask = inputs.to(self.device), labels.to(self.device), mask.to(self.device)
            
            noise = self.noise

            predictions = self.model(inputs * mask + (1 - mask) * noise)
            values = self.criterion(predictions, labels).reshape(-1, 1, 1, 1) * mask
            shapley_values_inf += values.sum(dim=0, keepdim=True)
            
            predictions = self.model(inputs * (1 - mask) + mask * noise)
            values = self.criterion(predictions, labels).reshape(-1, 1, 1, 1) * mask 
            shapley_values_sup += values.sum(dim=0, keepdim=True)

            masks += mask.sum(dim=0, keepdim=True)

        shapley_values_inf /= masks
        shapley_values_sup /= masks
        return shapley_values_inf, shapley_values_sup


class DeepExplainerModel(BaseExplanationModel):

    def __init__(self, model, criterion=None, noise_distribution=None, device="cuda"):

        super().__init__(model, criterion, noise_distribution, device)

    def fit(self, data_loader, n_samples=16, zero_baseline=True):
        
        self.model.eval()
        outputs = []

        example = next(iter(data_loader))[1]
        print("example", example.shape)

        self.explainer = shap.DeepExplainer(self.model, example.to(self.device))
        expected_value, predicted_labels = self.get_avg_output(data_loader)

        self.explainer.explainer.expected_value = expected_value
        self.explainer.expected_value = expected_value

        for i, data in enumerate(data_loader):
            _, img, __, ___ = data 

            shapley_score = self._fit(img)
            outputs.append((_, img.cpu(), __, shapley_score, predicted_labels[i].detach().cpu()))

            if i + 1 == n_samples:
                break
        return outputs

    def get_avg_output(self, data):
        with torch.no_grad():
            
            outputs, predicted_labels = [], []
            for d in data:
                _, img, __, ___ = d

                output = self.model(img.to(self.device)).cpu()
                outputs.append(output)
                predicted_labels.append(output.argmax(dim=-1))

            return torch.stack(outputs, dim=0).mean(dim=0), torch.stack(predicted_labels, dim=0)

    def _fit(self, sample):
        
        output = self.explainer.shap_values(sample.to(self.device))[0]
        output = torch.from_numpy(output).float()

        return output.mean(dim=1, keepdim=True)


