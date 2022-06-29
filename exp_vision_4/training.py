import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from math import sqrt
import numpy as np
import logging
from torchvision import models
from torch.optim import Adam


def load_model(path):

    model = models.vgg16(pretrained=True)

    try:
        model.classifier[6] = nn.Linear(4096, 2)
        model.load_state_dict(torch.load(path))
    except:
        model.classifier[6] = nn.Linear(4096, 37)
        model.load_state_dict(torch.load(path))
    return model

def build_model(train_config, train_loader, test_loader):

    model = models.vgg16(pretrained=train_config.from_pretrained)

    # freeze convolution weights
    if train_config.from_pretrained:
        for param in model.features.parameters():
            param.requires_grad = False

    # Change classifier
    if train_config.all_classes:
        model.classifier[6] = nn.Linear(4096, 37)
    else:
        model.classifier[6] = nn.Linear(4096, 2)
        
    model = model.to(train_config.device)
    optimizer = Adam(params=model.parameters(), lr=train_config.lr)
    criterion = nn.CrossEntropyLoss()
        
    trainer = Trainer(model, optimizer, criterion, device=train_config.device)
    trainer.train(train_loader, epochs=train_config.epochs)
    trainer.test(test_loader)

    del trainer 
    del optimizer 

    for param in model.features.parameters():
        param.requires_grad = False

    return model

class Trainer():

    def __init__(self, model, optimizer, criterion, device="cuda"):

        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, train_data, epochs=2):
        
        logging.info("Training model")
        self.model.train()

        counter = 0

        for epoch in range(epochs):
            cumulative_sum = 0
            for i, data in enumerate(train_data):

                self.optimizer.zero_grad()

                _, inputs, __, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                self.optimizer.step()

                cumulative_sum += loss.data / 100

                if (i + 1) % 100 == 0:
                    logging.info(f"epoch: {epoch + 1} step: {i + 1} loss: {cumulative_sum}")
                    cumulative_sum = 0

                counter += 1


    def test(self, test_data):
        
        with torch.no_grad():
            self.model.eval()

            accuracy = []
            for data in test_data:

                _, inputs, __, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                predicted = torch.argmax(outputs, dim=-1)
                accuracy.append((predicted == labels).squeeze().float())
            
            accuracy = torch.cat(accuracy).mean()
            logging.info(f"Test accuracy: {accuracy}")
            return accuracy

    def test_shapley(self, test_data, criterion=nn.CrossEntropyLoss(reduction="none"), num_samples=2, batch_size=32, threshold=150):
        
        self.model.eval()

        base_data = []
        shapley_data = []
        data = []

        for i, d in enumerate(test_data):
            data.append(d)
            if i + 1 == num_samples:
                break
        
        for sample in data:
            inputs, labels = sample
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            size = inputs.size()
            mask_size = size[-2] * size[-1]

            v_n = self.criterion(self.model(inputs), labels)

            prob, predicted_label = torch.softmax(self.model(inputs), dim=-1).max(dim=-1)
            v_n = - prob.log()
            

            mask = torch.zeros(mask_size, 1, mask_size)
            idx = torch.arange(mask_size).reshape(mask_size, 1, 1)
            mask = torch.scatter(mask, dim=-1, index=idx, src=torch.ones_like(idx).float())
            mask = mask.reshape(mask_size, 1, size[-2], size[-1])

            inf_inputs = inputs.detach().cpu() * mask
            sup_inputs = inputs.detach().cpu() * (1 - mask)
            
            inf_loader = DataLoader(TensorDataset(inf_inputs, predicted_label.expand(mask_size)), batch_size=batch_size)
            sup_loader = DataLoader(TensorDataset(sup_inputs, predicted_label.expand(mask_size)), batch_size=batch_size)

            # v(x)
            inf_values = self.shapley_loop(inf_loader, criterion)

            # v(K) - v(K\x)
            sup_values = v_n - self.shapley_loop(sup_loader, criterion)

            # w
            w = (v_n - sup_values.sum()) / (inf_values.sum() - sup_values.sum())
            
            # Shapley
            score = (w * inf_values + (1 - w) * sup_values)

            idx = score.argsort(dim=-1, descending=True)

            score_thresholed = torch.zeros_like(score)
            score_thresholed = torch.scatter(score_thresholed, dim=-1, index=idx[:threshold], src=torch.ones_like(score_thresholed)[:threshold])
            score_thresholed = torch.scatter(score_thresholed, dim=-1, index=idx[-threshold:], src=-torch.ones_like(score_thresholed)[-threshold:])

            #shapley_data.append((inputs[0, 0, :, :], w * inf_values + (1 - w) * sup_values))
            #shapley_data.append((labels.cpu().numpy(), score_thresholed.reshape(size[-2], size[-1]).detach().cpu().numpy()))
            #shapley_data.append((labels.cpu().numpy(), score_thresholed.reshape(size[-2], size[-1]).detach().cpu().numpy()))

            base_data.append((labels.cpu().numpy(), inputs[0, :, :, :].detach().cpu().numpy()))
            shapley_data.append((labels.cpu().numpy(), (inputs[0, :, :, :] + score_thresholed.reshape(size[-2], size[-1])*100).clamp(min=0, max=1).detach().cpu().numpy()))

        return shapley_data, base_data
        
    
    def test_shapley2(self, test_data, criterion=nn.CrossEntropyLoss(reduction="none"), num_samples=2, batch_size=32):
        
        self.model.eval()

        base_data = []
        shapley_data = []
        data = []

        for i, d in enumerate(test_data):
            data.append(d)
            if i + 1 == num_samples:
                break
        
        for sample in data:
            inputs, labels = sample
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            size = inputs.size()
            mask_size = size[-2] * size[-1]

            v_n = self.criterion(self.model(inputs), labels).detach().cpu().numpy()
            
            mask = torch.zeros(mask_size, 1, mask_size, device=self.device)
            idx = torch.arange(mask_size, device=self.device).reshape(mask_size, 1, 1)
            mask = torch.scatter(mask, dim=-1, index=idx, src=torch.ones_like(idx).float())
            mask = mask.reshape(mask_size, 1, size[-2], size[-1])

            inf_inputs = inputs * mask
            sup_inputs = inputs * (1 - mask)

            inf_loader = DataLoader(TensorDataset(inf_inputs, labels.expand(mask_size)), batch_size=batch_size)
            sup_loader = DataLoader(TensorDataset(sup_inputs, labels.expand(mask_size)), batch_size=batch_size)

            # v(x)
            inf_values = self.shapley_loop(inf_loader, criterion).reshape(size[-2], size[-1]).detach().cpu().numpy()

            # v(K) - v(K\x)
            sup_values = v_n - self.shapley_loop(sup_loader, criterion).reshape(size[-2], size[-1]).detach().cpu().numpy()

            A, B = inf_values.sum(), sup_values.sum()
            a, b, c = A**2 + B**2, -2*v_n*A, v_n**2 - B**2
            delta = b**2 - 4*a*c 

            # Case 1
            alpha_1 = (-b - sqrt(delta))/(2*a)
            alpha_2 = sqrt(1 - alpha_1**2)
            output = alpha_1*inf_values + alpha_2*sup_values

            if np.abs(np.sum(output)) > 1e-8:
                alpha_1 = (-b + sqrt(delta))/(2*a)
                alpha_2 = sqrt(1 - alpha_1**2)
                output = alpha_1*inf_values + alpha_2*sup_values

            #shapley_data.append((inputs[0, 0, :, :], w * inf_values + (1 - w) * sup_values))
            shapley_data.append((labels.cpu().numpy(), output))
            base_data.append((labels.cpu().numpy(), inputs[0, 0, :, :].detach().cpu().numpy()))
            #print(alpha_1, alpha_2)
        return shapley_data, base_data


    def shapley_loop(self, loader, criterion):
        
        losses = []
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            outputs = self.model(inputs)
            losses.append(criterion(outputs, labels))

            #losses.append(- torch.softmax(outputs, dim=-1).max(dim=-1)[0].log())
        return torch.cat(losses)