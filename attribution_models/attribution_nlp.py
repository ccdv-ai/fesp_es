import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class NLPAttributionModel(nn.Module):

    def __init__(self, model, tokenizer, batch_size=16, post_processing=None, device="cuda"):

        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device

        if post_processing is not None:
            self.post_processing = post_processing
        else:
            self.post_processing = lambda x: x

    def predict_function(self, pred):
        pred = pred["logits"]
        logits = self.post_processing(pred.detach()).cpu()
        prediction, argmax = logits.max(dim=-1, keepdim=True)
        return logits, prediction, argmax

    def single_prediction(self, inputs):
        outputs = self.model(
            input_ids=inputs["input_ids"].to(self.device), 
            attention_mask=inputs["attention_mask"].to(self.device)
            )
        logits, prediction, argmax = self.predict_function(outputs)
        return prediction, argmax

    def eval_batch(self, dataset, label):

        all_logits = []
        for input_ids, attention_mask in dataset:
            outputs = self.model(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
            logits, prediction, argmax = self.predict_function(outputs)
            all_logits.append(logits)

        return torch.cat(all_logits, dim=0)[:, int(label)].unsqueeze(-1)

    def get_dataset(self, *args):
        dataset = TensorDataset(*args)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False
            )

    def fit(self, samples, kernel=8, stride=1):
        
        """
        samples: list[str]
        """

        self.model = self.model.to(self.device)

        if isinstance(samples, str):
            samples = [samples]

        if not (isinstance(samples, list) or isinstance(samples, tuple)):
            raise "samples must be list[str] or tuple(str)"

        with torch.no_grad():

            self.model.eval()

            # Samples: list of samples
            outputs = []
            predicted_labels = []

            for sample in samples:
                dataset = self.build_mask(sample, kernel, stride)
                output, predicted_label = self._fit(sample, dataset)
                outputs.append(output)
                predicted_labels.append(int(predicted_label))

        self.model = self.model.cpu()
        return outputs, predicted_labels

    def build_mask(self, sample, kernel, stride):

        # Get subwords, ids and mask
        output = self.tokenizer(sample, return_special_tokens_mask=True, return_tensors ="pt", truncation=True, padding=True)

        # Get special tokens, they are not masked
        special_tokens_mask = output["special_tokens_mask"]
        num_tokens = (1 - special_tokens_mask).sum()
        split = self.tokenizer.tokenize(sample, truncation=True)[:num_tokens]
        filling_idx = (special_tokens_mask == 0).nonzero(as_tuple=False)

        mask = torch.ones(len(split))
        
        pad_size = kernel - stride

        # if kernel == 1 -> identity matrix
        if kernel > 1:

            pad = torch.zeros(pad_size)
            mask = torch.cat([pad, mask, pad])

            # Compute rolling window mask
            compressed_mask = []
            for i in range((mask.shape[0] - pad_size) // stride):
                copy_mask = torch.zeros_like(mask)
                copy_mask[i*stride:i*stride + kernel] = 1
                compressed_mask.append(copy_mask)
            compressed_mask = torch.stack(compressed_mask, dim=0)[:, pad_size:-pad_size]

        else:
            compressed_mask = torch.diag(torch.ones(mask.shape[0]))

        mask = special_tokens_mask.expand(compressed_mask.shape[0], -1)
        filling_idx = filling_idx[:, 1:].transpose(-1, -2).expand(mask.shape[0], -1)
        

        #print(mask.shape, filling_idx.shape, compressed_mask.shape)
        mask_reverse = mask.float().scatter(dim=-1, index=filling_idx, src=(1. - compressed_mask).float())
        mask = mask.float().scatter(dim=-1, index=filling_idx, src=compressed_mask.float())

        count = mask.sum(dim=0, keepdim=True)
        count_reverse = mask_reverse.sum(dim=0, keepdim=True)

        output["input_ids_reverse"] = output["input_ids"] * mask_reverse.long()
        output["attention_mask_reverse"] = mask_reverse.long()

        output["input_ids"] = output["input_ids"] * mask.long()
        output["attention_mask"] = mask.long()

        output["count"] = count
        output["count_reverse"] = count_reverse
        
        return output


class FESPWithTransformers(NLPAttributionModel):
    
    def __init__(self, model, tokenizer, batch_size=16, post_processing=None, device="cuda"):
        
        super().__init__(model, tokenizer, batch_size, post_processing, device)

    def _fit(self, sample, dataset):

        inputs = self.tokenizer(sample, return_tensors="pt", truncation=True, padding=True)
        pred, label = self.single_prediction(inputs)

        base_dataset = self.get_dataset(dataset["input_ids"], dataset["attention_mask"])
        base_outputs = self.eval_batch(base_dataset, label)
        base_loss = (base_outputs * dataset["attention_mask"]).sum(dim=0, keepdim=True)
        base_loss /= dataset["count"]

        reverse_dataset = self.get_dataset(dataset["input_ids_reverse"], dataset["attention_mask_reverse"])
        reverse_outputs = self.eval_batch(reverse_dataset, label)
        reverse_loss = (reverse_outputs * dataset["attention_mask"]).sum(dim=0, keepdim=True)
        reverse_loss /= dataset["count"]
        
        special_mask = 1 - dataset["special_tokens_mask"]

        w = (pred + reverse_loss.sum()) / (base_loss.sum() + reverse_loss.sum()) 
        scores = base_loss * w  - (1 - w) * reverse_loss
        scores = (scores * special_mask).squeeze(0)

        words = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        #return {"words": words, "scores": scores, "label": label[0]}
        return [(words[i], float(scores[i])) for i in range(len(words))], label
        

class ESWithTransformers(NLPAttributionModel):
    
    def __init__(self, model, tokenizer, batch_size=16, device="cuda"):
        
        super().__init__(model, tokenizer, batch_size, device)

    def _fit(self, sample, dataset):

        inputs = self.tokenizer(sample, return_tensors="pt", truncation=True, padding=True)
        pred, label = self.single_prediction(inputs)

        base_dataset = self.get_dataset(dataset["input_ids"], dataset["attention_mask"])
        base_outputs = self.eval_batch(base_dataset, label)
        base_loss = (base_outputs * dataset["attention_mask"]).sum(dim=0, keepdim=True)
        base_loss /= dataset["count"]
        
        special_mask = 1 - dataset["special_tokens_mask"]

        scores = base_loss + (pred - base_loss.sum()) / special_mask.shape[-1] 
        scores = (scores * special_mask).squeeze(0)

        words = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        #return {"words": words, "scores": scores, "label": label[0]}
        return [(words[i], float(scores[i])) for i in range(len(words))], label
        
