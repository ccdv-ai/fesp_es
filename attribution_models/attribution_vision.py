import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import math

class VisionAttributionModel(nn.Module):

    def __init__(self, model, noise_distribution=(0, 0), batch_size=64, post_processing=None, device="cuda"):

        super().__init__()

        self.model = model.eval()
        self.noise_distribution = noise_distribution

        self.batch_size = batch_size
        self.device = device

        if post_processing is not None:
            self.post_processing = post_processing
        else:
            self.post_processing = lambda x: x

    def predict_function(self, pred):

        logits = self.post_processing(pred.detach())
        prediction, argmax = logits.max(dim=-1)
        return logits, prediction, argmax

    def initial_step(self, sample):

        outputs = self.model(sample.to(self.device))
        logits, prediction, argmax = self.predict_function(outputs)
        return logits, prediction, argmax

    def get_custom_block_mask(self, image_shape=(3, 224, 224), block_size=(56, 56), stride=8, circular=True, balanced=False):
        
        if circular:
            assert image_shape[1] % block_size[0] == 0 and image_shape[2] % block_size[1] == 0

        c, h, w = image_shape
        bh, bw = block_size
        mask = torch.zeros(1, h, w)
        mask[:, :bh, :bw] = 1

        masks_ = []

        if not circular and not balanced:
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
        elif balanced:
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

    def set_distribution(self, img):
        n, c, h, w = img.size()
        self.noise = torch.normal(self.noise_distribution[0], self.noise_distribution[1], size=(n, c, h, w), device=self.device)
        
    def fit(self, imgs, block_size=(8, 8), stride=8, circular=False, balanced=False):
        
        '''
        imgs: tensor(N, C, H, W) or list[tensor(1, C, H, W)]
        '''

        if not isinstance(imgs, list):
            imgs = torch.split(imgs, 1, dim=0)

        mask = self.get_custom_block_mask(image_shape=imgs[0].shape[1:], block_size=block_size, stride=stride, circular=circular, balanced=balanced)

        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            outputs, labels = [], []

            for img in imgs:
                self.set_distribution(img)
                output, label = self._fit(img, mask)
                outputs.append(output.cpu())
                labels.append(label.cpu())

        self.model = self.model.cpu()
        return torch.cat(outputs, dim=0), torch.cat(labels, dim=0)

    def get_top_scores(self, scores, top=0.15):
        
        n, c, h, w = scores.size()
        scores = scores.reshape(n*c, h*w)
        idx = scores.argsort(dim=-1, descending=True)
        idx = idx[:, :int(top * h * w)]

        mask = torch.zeros_like(scores).scatter(dim=-1, index=idx, src=torch.ones_like(idx).float())

        return mask.reshape(n, c, h, w)

    def smooth_scores(self, attributions, param=0.95, size=9):

        from scipy.ndimage import convolve

        size = 9
        param = 0.95
        a = torch.arange(size).reshape(1, size)
        filtr = (param**(a + a.T)).roll((size//2, size//2), (-1, -2)).unsqueeze(0).unsqueeze(0)
        attributions = convolve(attributions, filtr.numpy(), mode='constant')
        return torch.from_numpy(attributions)


class FESPForVision(VisionAttributionModel):

    def __init__(self, model, noise_distribution=(0, 0), batch_size=64, post_processing=None, device="cuda"):

        super().__init__(model, noise_distribution, batch_size, post_processing, device)

    def _fit(self, sample, masks):
        
        logits, v_n, predicted_label = self.initial_step(sample)
        self.set_distribution(sample)
        scores = 0
        masks_sum = 0

        k = masks[0][0, 0].sum()
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

            w = (v_n - sup_sum) / (inf_sum - sup_sum)
            scores += w * inf_values + (1 - w) * sup_values
            masks_sum += mask.sum(dim=0, keepdim=True)

        scores = scores.cpu() / len(mask)
        return (scores, predicted_label)

    def shapley_loop(self, data):

        masks = 1e-6
        shapley_values_inf = 0
        shapley_values_sup = 0

        for batch in data:
            inputs, labels, mask = batch
            inputs, labels, mask = inputs.to(self.device), labels.to(self.device), mask.to(self.device)

            predictions = self.model(inputs * mask + (1 - mask) * self.noise)
            logits, _, _ = self.predict_function(predictions)
            values = logits[:, labels[0]].reshape(-1, 1, 1, 1) * mask
            shapley_values_inf += values.sum(dim=0, keepdim=True)
            
            predictions = self.model(inputs * (1 - mask) + mask * self.noise)
            logits, _, _ = self.predict_function(predictions)
            values = logits[:, labels[0]].reshape(-1, 1, 1, 1) * mask
            shapley_values_sup += values.sum(dim=0, keepdim=True)

            masks += mask.sum(dim=0, keepdim=True)

        shapley_values_inf /= masks + 1e-8
        shapley_values_sup /= masks + 1e-8
        return shapley_values_inf, shapley_values_sup


class ESForVision(VisionAttributionModel):

    def __init__(self, model, noise_distribution=(0, 0), batch_size=64, post_processing=None, device="cuda"):

        super().__init__(model, noise_distribution, batch_size, post_processing, device)

    def _fit(self, sample, masks):

        logits, v_n, predicted_label = self.initial_step(sample)
        self.set_distribution(sample)
        scores = 0

        k = masks[0][0, 0].sum()
        for mask in masks:
            n = mask.shape[0]
            loader = DataLoader(
                TensorDataset(sample.expand(mask.shape[0], -1, -1, -1), predicted_label.expand(n).cpu(), mask), 
                batch_size=self.batch_size
                ) 

            inf_values = self.shapley_loop(loader)
            inf_values = inf_values.mean(dim=1, keepdim=True)

            h, w = inf_values.shape[-2:]
            scores += inf_values + (v_n - inf_values.sum()) / (h*w)

        scores = scores / len(masks)
        return (scores, predicted_label)

    def shapley_loop(self, data):

        masks = 1e-6
        shapley_values_inf = 0

        for batch in data:
            inputs, labels, mask = batch
            inputs, labels, mask = inputs.to(self.device), labels.to(self.device), mask.to(self.device)

            predictions = self.model(inputs * mask + (1 - mask) * self.noise)
            logits, _, _ = self.predict_function(predictions)
            values = logits[:, labels[0]].reshape(-1, 1, 1, 1) * mask
            shapley_values_inf += values.sum(dim=0, keepdim=True)

            masks += mask.sum(dim=0, keepdim=True)

        shapley_values_inf /= masks
        return shapley_values_inf


class OcclusionForVision(VisionAttributionModel):

    def __init__(self, model, noise_distribution=(0, 0), batch_size=64, post_processing=None, device="cuda"):

        super().__init__(model, noise_distribution, batch_size, post_processing, device)

    def _fit(self, sample, masks):

        logits, v_n, predicted_label = self.initial_step(sample)
        self.set_distribution(sample)
        scores = 0
        
        k = masks[0][0, 0].sum()
        for mask in masks:
            n = mask.shape[0]
            loader = DataLoader(
                TensorDataset(sample.expand(mask.shape[0], -1, -1, -1), predicted_label.expand(n).cpu(), mask), 
                batch_size=self.batch_size
                ) 

            sup_values = self.shapley_loop(loader)
            sup_values = v_n - sup_values.mean(dim=1, keepdim=True)
            scores += sup_values

        scores = scores / len(masks)
        return (scores, predicted_label)

    def shapley_loop(self, data):

        masks = 1e-6
        shapley_values_sup = 0

        for batch in data:
            inputs, labels, mask = batch
            inputs, labels, mask = inputs.to(self.device), labels.to(self.device), mask.to(self.device)

            predictions = self.model(inputs * (1 - mask) + mask * self.noise)
            logits, _, _ = self.predict_function(predictions)
            values = logits[:, labels[0]].reshape(-1, 1, 1, 1) * mask
            shapley_values_sup += values.sum(dim=0, keepdim=True)

            masks += mask.sum(dim=0, keepdim=True)

        shapley_values_sup /= masks
        return shapley_values_sup

