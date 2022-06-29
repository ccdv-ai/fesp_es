import torch
from torch.utils.data import TensorDataset, DataLoader
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    Occlusion
)

import shap
from transformers import pipeline


class BaseAttributionModel(object):

    def __init__(self, model, tokenizer, sep_token=None, device="cuda"):
        
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        if sep_token is None:
            self.sep_token = self.tokenizer.tokenize("1 2")[1][0]

    def merge_scores(self, scores, f, whole_word=False):
        
        outputs = []
        if whole_word:
            
            for score_list in scores:
                
                sentence = "".join(score_list["words"]).replace(self.sep_token, " ")
                current_word_list = []
                current_score_list = []

                for i, (word, score) in enumerate(zip(score_list["words"], score_list["scores"])):
                    
                    if word[0] == self.sep_token or i == 0:

                        if i > 0:
                            current_score_list.append(f(s))
                            current_word_list.append(w)

                        s = [score]
                        w = word
                    elif word[0] != self.sep_token:
                        s.append(score)
                        w += word

                current_score_list.append(f(s))
                current_word_list.append(w)

                current_word_list = [w.replace(self.sep_token, "") for w in current_word_list]
                #current_word_list = [w for w in current_word_list]
                outputs.append({"words": current_word_list, "scores": current_score_list, "sentence": sentence})
            return outputs
        else:
            for score_list in scores:
                sentence = "".join(score_list["words"]).replace(self.sep_token, " ")
                outputs.append({
                    "words": [w.replace(self.sep_token, "") for w in score_list["words"]], 
                    "scores": [s for s in score_list["scores"]], 
                    "sentence": sentence}
                    )
            return outputs
        
    def save_to_file(self, path, scores, labels, dataset_name=None):
        
        import json
        
        for score in scores:
            score["scores"] = [float(s) for s in score["scores"]]
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"name": dataset_name, "scores": scores, "labels": [int(l) for l in labels]}, f, indent=4)

    def load_file(self, path):
        import json
        with open(path, "r", encoding="utf-8") as f:
            scores = json.load(f)

        return scores["scores"], torch.tensor(scores["labels"]).clone().detach(), scores["name"]

    def plot_for_notebook(self, scores):

        from IPython import display as d
        from IPython.core.display import HTML
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(feature_range=(0, 255))

        all_html = ""
        for output in scores:
            print("\n")
            s = ""
            
            scores = torch.tensor(output["scores"]).clone().detach()
            scores = scaler.fit_transform(scores.unsqueeze(-1).numpy()).reshape(-1)

            for word, score in zip(output["words"], scores):
                s += f"<span style='background-color:rgb({score} , 140, 140 );'>{word} </span>"
            all_html = all_html + s + "<br>"
        return all_html


class FESPWithTransformers(BaseAttributionModel):
    
    def __init__(self, model, tokenizer, batch_size=16, sep_token=None, predict_function=None, device="cuda"):
        
        super().__init__(model, tokenizer, sep_token, device)
            
        self.batch_size = batch_size

        if predict_function is not None:
            self.predict_function = predict_function

    def predict_function(self, pred):
        return torch.softmax(pred["logits"].detach(), dim=-1).cpu()

    def get_dataset(self, *args):
        dataset = TensorDataset(*args)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False
            )

    def fit(self, samples, kernel=8, stride=1, whole_word=False):

        # Samples: list of samples
        outputs = []
        for sample in samples:
            dataset = self.build_mask(sample, kernel, stride, whole_word)
            outputs.append(self._fit(sample, dataset))
        self.model = self.model.cpu()
        return outputs

    def _fit(self, sample, dataset):
        
        self.model.eval()
        inputs = self.tokenizer(sample, return_tensors="pt", truncation=True, padding=True)
        pred = self.model(input_ids=inputs["input_ids"].to(self.device), attention_mask=inputs["attention_mask"].to(self.device))
        pred = self.predict_function(pred)
        pred, argmax = pred.max(dim=-1, keepdim=True)

        base_dataset = self.get_dataset(dataset["input_ids"], dataset["attention_mask"])
        reverse_dataset = self.get_dataset(dataset["input_ids_reverse"], dataset["attention_mask_reverse"])

        base_outputs = self.eval_batch(base_dataset)
        reverse_outputs = self.eval_batch(reverse_dataset)
        argmax = argmax.expand(base_outputs.shape[0], -1)

        base_loss = (base_outputs.gather(dim=-1, index=argmax) * dataset["attention_mask"]).sum(dim=0, keepdim=True)
        base_loss /= dataset["count"]

        reverse_loss = (reverse_outputs.gather(dim=-1, index=argmax) * dataset["attention_mask"]).sum(dim=0, keepdim=True)
        reverse_loss /= dataset["count"]
        
        mask = 1 - dataset["special_tokens_mask"]

        w = (pred + reverse_loss.sum()) / (base_loss.sum() + reverse_loss.sum()) 

        score = base_loss * w  - (1 - w) * reverse_loss
        score *= 1 - dataset["special_tokens_mask"]

        scores = torch.tensor([s for s in score.reshape(-1) if s != 0])
        #scores = torch.softmax(torch.tensor(scores), dim=-1)
        words = self.tokenizer.tokenize(sample, truncation=True)[:mask.sum()]

        assert len(words) == len(scores)
        return {"words": words, "scores": scores}

    def eval_batch(self, dataset):

        self.model.eval()
        preds = []
        for input_ids, attention_mask in dataset:
            pred = self.model(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
            pred = self.predict_function(pred)
            preds.append(pred)

        return torch.cat(preds, dim=0)

    def build_mask(self, sample, kernel, stride, whole_word):

        # Get subwords, ids and mask
        output = self.tokenizer(sample, return_special_tokens_mask=True, return_tensors ="pt", truncation=True, padding=True)

        # Get special tokens, they are not masked
        special_tokens_mask = output["special_tokens_mask"]
        num_tokens = (1 - special_tokens_mask).sum()

        split = self.tokenizer.tokenize(sample, truncation=True)[:num_tokens]

        filling_idx = (special_tokens_mask == 0).nonzero(as_tuple=False)


        # Count in case of whole word
        whole_word_mask = []
        whole_word_count = []
        count = 1

        for i, word in enumerate(split):
            if word[0] == self.sep_token or i == 0:
                if i > 0:
                    whole_word_count.append(count)
                count = 1
                whole_word_mask.append(1)
            elif word[0] != self.sep_token:
                count += 1
        whole_word_count.append(count)


        if not whole_word:
            mask = torch.ones(len(split))
        else:
            mask = torch.tensor(whole_word_mask)
        

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

        # Duplicate entries if whole words
        if whole_word:
            outputs = []
            for i, count in enumerate(whole_word_count):
                outputs.append(compressed_mask[:, i].reshape(-1, 1).expand(-1, count))

            compressed_mask = torch.cat(outputs, dim=-1)

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

    def process_scores(self, scores, method="avg", top=0.15):
        
        assert method in ["avg", "top"]

        if method == "top":
            for sentence in scores:
                size = int(top * len(sentence["words"]))
                s = sentence["scores"]
                idx = s.argsort(dim=-1, descending=True)[:size]
                s = torch.zeros_like(s).float().scatter(dim=-1, index=idx, src=torch.ones_like(idx).float())
                sentence["top_scores"] = s

        elif method == "avg":
            for sentence in scores:
                s = sentence["scores"]
                s = (torch.sign(s - s.mean()) + 1)/2
                sentence["top_scores"] = s
        
        return scores


class ESWithTransformers(BaseAttributionModel):
    
    def __init__(self, model, tokenizer, batch_size=16, sep_token=None, predict_function=None, device="cuda"):
        
        super().__init__(model, tokenizer, sep_token, device)
            
        self.batch_size = batch_size

        if predict_function is not None:
            self.predict_function = predict_function

    def predict_function(self, pred):
        return torch.softmax(pred["logits"].detach(), dim=-1).cpu()

    def get_dataset(self, *args):
        dataset = TensorDataset(*args)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False
            )

    def fit(self, samples, kernel=8, stride=1, whole_word=False):

        # Samples: list of samples
        outputs = []
        for sample in samples:
            dataset = self.build_mask(sample, kernel, stride, whole_word)
            outputs.append(self._fit(sample, dataset))
        return outputs

    def _fit(self, sample, dataset):
        
        self.model.eval()
        inputs = self.tokenizer(sample, return_tensors="pt", truncation=True, padding=True)
        pred = self.model(input_ids=inputs["input_ids"].to(self.device), attention_mask=inputs["attention_mask"].to(self.device))
        pred = self.predict_function(pred)
        pred, argmax = pred.max(dim=-1, keepdim=True)
        pred = 1

        base_dataset = self.get_dataset(dataset["input_ids"], dataset["attention_mask"])
        reverse_dataset = self.get_dataset(dataset["input_ids_reverse"], dataset["attention_mask_reverse"])

        base_outputs = self.eval_batch(base_dataset)
        argmax = argmax.expand(base_outputs.shape[0], -1)

        base_loss = (base_outputs.gather(dim=-1, index=argmax) * dataset["attention_mask"]).sum(dim=0, keepdim=True)
        base_loss /= dataset["count"]

        #reverse_loss = (reverse_outputs.gather(dim=-1, index=argmax) * dataset["attention_mask_reverse"]).sum(dim=0, keepdim=True)
        #reverse_loss /= dataset["count_reverse"]
        

        #w = (pred + reverse_loss.sum()) / (base_loss.sum() + reverse_loss.sum()) 

        mask = 1 - dataset["special_tokens_mask"]
        #base_loss *= mask
        score = base_loss + (pred - base_loss.sum()) / mask.shape[-1]
        score *= 1 - dataset["special_tokens_mask"]

        scores = torch.tensor([s for s in score.reshape(-1) if s != 0])
        #scores = torch.softmax(torch.tensor(scores), dim=-1)
        words = self.tokenizer.tokenize(sample, truncation=True)[:mask.sum()]

        assert len(words) == len(scores)
        return {"words": words, "scores": scores}

    def eval_batch(self, dataset):

        self.model.eval()
        preds = []
        for input_ids, attention_mask in dataset:
            pred = self.model(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
            pred = self.predict_function(pred)
            preds.append(pred)

        return torch.cat(preds, dim=0)

    def build_mask(self, sample, kernel, stride, whole_word):

        # Get subwords, ids and mask
        output = self.tokenizer(sample, return_special_tokens_mask=True, return_tensors ="pt", truncation=True, padding=True)

        # Get special tokens, they are not masked
        special_tokens_mask = output["special_tokens_mask"]
        num_tokens = (1 - special_tokens_mask).sum()

        split = self.tokenizer.tokenize(sample, truncation=True)[:num_tokens]

        filling_idx = (special_tokens_mask == 0).nonzero(as_tuple=False)


        # Count in case of whole word
        whole_word_mask = []
        whole_word_count = []
        count = 1

        for i, word in enumerate(split):
            if word[0] == self.sep_token or i == 0:
                if i > 0:
                    whole_word_count.append(count)
                count = 1
                whole_word_mask.append(1)
            elif word[0] != self.sep_token:
                count += 1
        whole_word_count.append(count)


        if not whole_word:
            mask = torch.ones(len(split))
        else:
            mask = torch.tensor(whole_word_mask)
        

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

        # Duplicate entries if whole words
        if whole_word:
            outputs = []
            for i, count in enumerate(whole_word_count):
                outputs.append(compressed_mask[:, i].reshape(-1, 1).expand(-1, count))

            compressed_mask = torch.cat(outputs, dim=-1)

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

    def process_scores(self, scores, method="avg", top=0.15):
        
        assert method in ["avg", "top"]

        if method == "top":
            for sentence in scores:
                size = int(top * len(sentence["words"]))
                s = sentence["scores"]
                idx = s.argsort(dim=-1, descending=True)[:size]
                s = torch.zeros_like(s).float().scatter(dim=-1, index=idx, src=torch.ones_like(idx).float())
                sentence["top_scores"] = s

        elif method == "avg":
            for sentence in scores:
                s = sentence["scores"]
                s = (torch.sign(s - s.mean()) + 1)/2
                sentence["top_scores"] = s
        
        return scores


class CaptumWrapperWithTransformers(BaseAttributionModel):

    def __init__(self, model, captum_model, tokenizer, embedding_layer=None, sep_token=None, device="cuda"):
        
        super().__init__(model, tokenizer, sep_token, device)
        
        self.captum_model = captum_model

        if embedding_layer is not None:
            self.embedding_layer = embedding_layer
        else:
            self.embedding_layer = self.model.roberta.embeddings

    def fit(self, samples, kernel=1, stride=1):
        
        results = []
        for sample in samples:

            results.append(self._fit(sample, kernel=kernel, stride=stride))
        self.model = self.model.cpu()
        return results
    
    def _fit(self, sample, kernel, stride):

        tokenized_input = self.tokenizer(sample, return_tensors="pt", return_special_tokens_mask=True, padding=True, truncation=True)
        special_tokens = tokenized_input["special_tokens_mask"].to(self.device)
        num_tokens = (1 - special_tokens).sum()

        inputs = self.embedding_layer(tokenized_input["input_ids"].to(self.device))#.detach()
        #inputs.requires_grad = True

        baselines = torch.zeros_like(inputs)

        target = self.model(inputs.to(self.device)).argmax(dim=-1)
        
        #print(inputs.shape, baselines.shape)
        #attributions, delta = self.captum_model.attribute(inputs, baselines, target=target, return_convergence_delta=True)

        if type(self.captum_model).__name__ == "Occlusion":
            attributions = self.captum_model.attribute(
                inputs, 
                sliding_window_shapes=(kernel, 768), 
                strides=stride,
                baselines=baselines, 
                target=target, 
                )

        elif type(self.captum_model).__name__ == "IntegratedGradients":
            attributions, delta = self.captum_model.attribute(
                inputs, 
                baselines=baselines, 
                target=target, 
                return_convergence_delta=True, 
                internal_batch_size=16
                )
        elif type(self.captum_model).__name__ == "GradientShap" or type(self.captum_model).__name__ == "DeepLift":
            attributions, delta = self.captum_model.attribute(
                inputs, 
                baselines=baselines, 
                target=target, 
                return_convergence_delta=True, 
                )

        else:
            attributions = self.captum_model.attribute(
                inputs, 
                strides=1,
                sliding_window_shapes=1,
                baselines=baselines, 
                target=target, 
                )

        attributions = (attributions.sum(dim=-1) * (1 - special_tokens)).reshape(-1).cpu()
        attributions /= attributions.norm()
        
        outputs = {"words": self.tokenizer.tokenize(sample, truncation=True)[:num_tokens], "scores": [float(a) for a in attributions if a != 0]}

        del attributions
        del inputs

        idx = min(len(outputs["words"]), len(outputs["scores"]))
        outputs["words"] = outputs["words"][:idx]
        outputs["scores"] = outputs["scores"][:idx]
        assert len(outputs["words"]) == len(outputs["scores"])
        return outputs


class ShapExplainerWrapperWithTransformers(BaseAttributionModel):

    def __init__(self, model, tokenizer, sep_token=None):
        
        super().__init__(model, tokenizer, sep_token, device="cpu")

        self.pipeline = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
        #self.shap_explainer = shap.Explainer(self.pipeline, algorithm="permutation")
        self.shap_explainer = shap.Explainer(self.pipeline, algorithm="auto")
        
    def fit(self, samples):
        
        results = []
        for sample in samples:
            label = self.eval_sample(sample)
            results.append(self._fit(sample, label))
        self.model = self.model.cpu()
        return results
        
    def _fit(self, sample, label):

        sample = "".join(self.tokenizer.tokenize(sample, truncation=True)[:self.max_size]).replace(self.sep_token, " ")

        outputs = self.shap_explainer([sample])
        values = outputs.values[..., label][0]
        text_base = outputs.data[0]
        text_tokenized = self.tokenizer.tokenize(sample, truncation=True)[:self.max_size]

        count = 0
        words, scores = [], []
        for i, word in enumerate(text_base):
            if word != "":
                words.append(text_tokenized[count])
                scores.append(values[i])
                count += 1
                
            if (word == "" and count > 0) or count == len(text_tokenized):
                break

        return {"words": words, "scores": scores}
        
    def eval_sample(self, sample):
        
        inputs = self.tokenizer(sample, return_tensors="pt", padding=True, truncation=True, return_special_tokens_mask=True)
        label = self.model(inputs["input_ids"], inputs["attention_mask"])["logits"].argmax(dim=-1)
        self.max_size = int((1 - inputs["special_tokens_mask"]).sum())
        self.num_special_tokens = int(inputs["special_tokens_mask"].sum())
        return label
    

class EvalAttributionModelWithTransformers:

    def __init__(self, model, tokenizer, reversed=False, device="cuda"):

        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.sep_token = self.tokenizer.tokenize("1 2")[1][0]
        self.reversed = reversed
        self.device = device

    def predict_function(self, pred):
        return torch.softmax(pred["logits"].detach(), dim=-1).cpu()

    def fit(self, scores, labels, top=0.10, descending=True):
        
        outputs = []
        self.model.eval()
        for i, score in enumerate(scores):

            input_ids, mask = self.build_dataset(score, top, descending)

            pred = self.predict_function(self.model(input_ids.to(self.device)))
            pred, argmax = pred.max(dim=-1)

            output = self.predict_function(self.model(input_ids.to(self.device), mask.to(self.device)))
            output = (labels[i] == output.argmax(dim=-1)).float().reshape(-1)
            outputs.append(output.cpu().detach())

        acc = torch.tensor(outputs)
        return acc.mean()

    def build_dataset(self, score, top, descending):

        #print(scores)
        words, s = score["words"], torch.tensor(score["scores"])

        if top == 1:
            mask = torch.ones_like(s)
        elif top == 0:
            mask = torch.zeros_like(s)
        else:
            n = round(s.shape[0] * top)
            idx = s.argsort(dim=-1, descending=descending)[:n]
            mask = torch.zeros_like(s)
            mask[idx] = 1
            #print("1", mask.shape)
            mask = [m for m in mask]
        
        
        sentence = score["sentence"]

        output = self.tokenizer(sentence, return_special_tokens_mask=True, return_tensors ="pt", padding=True, truncation=True)
        special_tokens_mask = output["special_tokens_mask"]

        tokenized_sentence = self.tokenizer.tokenize(sentence, truncation=True)

        
        if len(words) != len(tokenized_sentence):
            count = -1
            sentence_mask = []
            for i, word in enumerate(tokenized_sentence): 
                
                if i == 0 or word[0] == self.sep_token:
                    count += 1
                sentence_mask.append(mask[count])
        else:
            sentence_mask = mask
        
        #print(len(words), len(tokenized_sentence), len(sentence_mask))

        

        # Get special tokens, they are not masked
        filling_idx = (special_tokens_mask == 0).nonzero(as_tuple=False)[:, 1:].reshape(1, -1)

        if not self.reversed:
            mask = special_tokens_mask.float().scatter(dim=-1, index=filling_idx, src=torch.tensor(sentence_mask).unsqueeze(0).float())
        else:
            mask = special_tokens_mask.float().scatter(dim=-1, index=filling_idx, src=1 - torch.tensor(sentence_mask).unsqueeze(0).float())

        return output["input_ids"], mask

    def load_file(self, path):
        import json
        with open(path, "r", encoding="utf-8") as f:
            scores = json.load(f)

        return scores["scores"], torch.tensor(scores["labels"]).clone().detach(), scores["name"]

