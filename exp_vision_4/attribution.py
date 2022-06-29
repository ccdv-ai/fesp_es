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
        
        self.model = model.to(device).eval()
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
        #label = outputs.argmin(dim=-1)
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

    def get_custom_block_mask(self, image_shape=(3, 224, 224), block_size=(56, 56), stride=8, full_mask_only=False, circular=False, smoothing_degree=None):
        
        if circular:
            assert image_shape[1] % block_size[0] == 0 and image_shape[2] % block_size[1] == 0

        c, h, w = image_shape
        bh, bw = block_size
        mask = torch.zeros(1, h, w)
        mask[:, :bh, :bw] = 1

        masks_ = []

        if full_mask_only:
            for i in range((h - bh)//stride + 1):
                for j in range((w - bw)//stride + 1):
                    masks_.append(mask.roll((stride*i, stride*j), dims=(-2, -1)))
            return [torch.stack(masks_, dim=0)]

        # Circular
        elif circular:
            masks = []
            for i in range(h//bh):
                for j in range(w//bw):
                    masks.append(mask.roll((bh*i, bw*j), dims=(-2, -1)))
            masks = torch.stack(masks, dim=0)   

            for i in range(bh//stride):
                for j in range(bw//stride):
                    m = masks.roll((stride*i, stride*j), dims=(-2, -1))
                    masks_.append(m)
            return masks_
        else:
            masks = []
            mask = torch.zeros(1, h, w)
            for i in range(h // stride + block_size[0]//stride - 1):
                for j in range(w // stride + block_size[1]//stride - 1):
                    new_mask = mask.clone()
                    i_0 = max(-block_size[0] + stride*(i + 1), 0)
                    i_1 = min(stride*(i + 1), h)

                    j_0 = max(-block_size[1] + stride*(j + 1), 0)
                    j_1 = min(stride*(j + 1), h)
                    new_mask[:, i_0:i_1, j_0:j_1] = 1
                    masks.append(new_mask)
            return [torch.stack(masks, dim=0)]

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
    
    
class IntegratedGradientsModel(BaseExplanationModel):

    def __init__(self, model, criterion=None, noise_distribution=None, device="cuda"):

        super().__init__(model, criterion, noise_distribution, device)

    def fit(self, data_loader, n_samples=16, zero_baseline=True):
        
        self.model.eval()
        outputs = []

        for i, data in enumerate(data_loader):
            _, img, __, ___, ____ = data 

            shapley_score, label = self._fit(_, img, zero_baseline)
            outputs.append((_, img.cpu(), __, shapley_score.detach().cpu(), label.detach().cpu(), ____.cpu()))

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
            _, img, __, ___, ____ = data 

            shapley_score, label = self._fit(_, img, n_reestimations, zero_baseline)
            outputs.append((_, img.cpu(), __, shapley_score.detach().cpu(), label.detach().cpu(), ____.cpu()))

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
            _, img, __, ___, ____ = data 

            shapley_score, label = self._fit(_, img, zero_baseline)
            outputs.append((_, img.cpu(), __, shapley_score.detach().cpu(), label.detach().cpu(), ____.cpu()))

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
            _, img, __, ___, ____ = data 
            d.append(img)
            if i == size - 1:
                break
        return torch.cat(d, dim=0)

    def fit(self, data_loader, n_samples=16, zero_baseline=True, baseline_size=16):
        
        self.model.eval()

        d = self.get_distribution(data_loader, baseline_size)

        outputs = []
        for i, data in enumerate(data_loader):
            _, img, __, ___, ____ = data 

            shapley_score, label = self._fit(d, img, zero_baseline)
            outputs.append((_, img.cpu(), __, shapley_score.detach().cpu(), label.detach().cpu(), ____.cpu()))

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
            _, img, __, ___, ____ = data 

            shapley_score, label = self._fit(_, img, strides, sliding_window_shapes)
            outputs.append((_, img.cpu(), __, shapley_score.detach().cpu(), label.detach().cpu(), ____.cpu()))

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
            _, img, __, ___, ____ = data 

            shapley_score, label = self._fit(img, mask)

            outputs.append((_, img.cpu(), __, shapley_score.detach().cpu(), label.detach().cpu(), ____.cpu()))

            if i + 1 == n_samples:
                break
        return outputs

    def _fit(self, sample, masks):

        v_n, predicted_label = self.initial_step(sample)
        scores = 0

        self.set_distribution(sample)

        for mask in masks:
            n = mask.shape[0]
            loader = DataLoader(
                TensorDataset(sample.expand(mask.shape[0], -1, -1, -1), predicted_label.expand(n).cpu(), mask), 
                batch_size=self.batch_size
                ) 

            with torch.no_grad():
                inf_values = self.shapley_loop(loader)

            h, w = inf_values.shape[-2:]
            inf_values = inf_values.mean(dim=1, keepdim=True)

            scores += inf_values + (v_n - inf_values.sum()) / (h*w)

        scores = scores / len(masks)
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
            _, img, __, ___, ____ = data 

            shapley_score, label = self._fit(img, mask)
            outputs.append((_, img.cpu(), __, shapley_score.detach().cpu(), label.detach().cpu(), ____.cpu()))

            if i + 1 == n_samples:
                break
        return outputs

    def _fit(self, sample, masks):

        v_n, predicted_label = self.initial_step(sample)
        scores = 0

        self.set_distribution(sample)

        for mask in masks:
            n = mask.shape[0]
            loader = DataLoader(
                TensorDataset(sample.expand(mask.shape[0], -1, -1, -1), predicted_label.expand(n).cpu(), mask), 
                batch_size=self.batch_size
                ) 

            inf_values, sup_values = self.shapley_loop(loader)
            inf_values, sup_values = inf_values.mean(dim=1, keepdim=True), sup_values.mean(dim=1, keepdim=True)
            h, w = sample.shape[-2:]
            sup_values = (0 - sup_values)

            inf_sum = inf_values.sum()
            sup_sum = sup_values.sum()
            #inf_sum = inf_values.mean() * n
            #sup_sum = sup_values.mean() * n
            
            w = (v_n - sup_sum) / (inf_sum - sup_sum)
            scores += w * inf_values + (1 - w) * sup_values

        scores = scores / len(masks)
        return (scores, predicted_label)

    def shapley_loop(self, data):

        inputs, labels, mask = next(iter(data))

        masks = 1e-6
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

        self.explainer = shap.DeepExplainer(self.model, example.to(self.device))
        expected_value, predicted_labels = self.get_avg_output(data_loader)

        self.explainer.explainer.expected_value = expected_value
        self.explainer.expected_value = expected_value

        for i, data in enumerate(data_loader):
            _, img, __, ___, ____ = data 

            shapley_score = self._fit(img)
            outputs.append((_, img.cpu(), __, shapley_score, predicted_labels[i].detach().cpu(), ____.cpu()))

            if i + 1 == n_samples:
                break
        return outputs

    def get_avg_output(self, data):
        with torch.no_grad():
            
            outputs, predicted_labels = [], []
            for d in data:
                _, img, __, ___, ____ = d

                output = self.model(img.to(self.device)).cpu()
                outputs.append(output)
                predicted_labels.append(output.argmax(dim=-1))

            return torch.stack(outputs, dim=0).mean(dim=0), torch.stack(predicted_labels, dim=0)

    def _fit(self, sample):
        
        output = self.explainer.shap_values(sample.to(self.device))[0]
        output = torch.from_numpy(output).float()

        return output.mean(dim=1, keepdim=True)


