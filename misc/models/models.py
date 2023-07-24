import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import torchxrayvision as xrv
import torchvision
from ray.air import session
from train.params import Hparams
from torchvision import models
"""

"""
class DenseNetXRVAdversarial(pl.LightningModule):
    def __init__(self, args: Hparams, confusion = None):
        super().__init__()
        self.model_name = args.model_name
        self.num_classes_disease = args.num_classes_disease
        self.num_classes_sex = args.num_classes_sex
        self.num_classes_race = args.num_classes_race
        self.alpha = args.alpha
        self.fading_in_steps = args.fading_in_steps
        self.fading_in_range = args.fading_in_range
        self.confusion = confusion
        self.validation_step_outputs = []

        self.lr_d = args.lr_d
        self.lr_s = args.lr_s
        self.lr_r = args.lr_r
        self.lr_b = args.lr_b

        self.class_weights_race = torch.FloatTensor(tuple(args.class_weights_race))

        if self.model_name == 'imagenet':
            self.backbone = models.densenet121(pretrained=True)
        else:
            self.backbone = xrv.models.DenseNet(weights=f"densenet121-res224-{self.model_name}")
            self.backbone.op_threshs = None

        num_features = self.backbone.classifier.in_features
        self.fc_disease = nn.Linear(num_features, self.num_classes_disease)
        self.fc_sex = nn.Linear(num_features, self.num_classes_sex)
        self.fc_race = nn.Linear(num_features, self.num_classes_race)
        self.fc_connect = nn.Identity(num_features)
        self.backbone.classifier = self.fc_connect

    def forward(self, x):
        embedding = self.backbone.forward(x)
        out_disease = self.fc_disease(embedding)
        out_sex = self.fc_sex(embedding)
        out_race = self.fc_race(embedding)
        return out_disease, out_sex, out_race

    def configure_optimizers(self):
        params_backbone = list(self.backbone.parameters())
        params_disease = params_backbone + list(self.fc_disease.parameters())
        
        if self.confusion is None:
            params_sex, params_race = (params_backbone + list(self.fc_sex.parameters()), 
                                       params_backbone + list(self.fc_race.parameters()))
        else: 
            params_sex, params_race = list(self.fc_sex.parameters()), list(self.fc_race.parameters()) 
            
        
        optim_backbone = torch.optim.Adam(params_backbone, lr=self.lr_b)
        optim_disease = torch.optim.Adam(params_disease, lr=self.lr_d)
        optim_sex = torch.optim.Adam(params_sex, lr=self.lr_s)
        optim_race = torch.optim.Adam(params_race, lr=self.lr_r)
        return optim_disease, optim_sex, optim_race, optim_backbone

    def unpack_batch(self, batch):
        return batch['image'], batch['label_disease'], batch['label_sex'], batch['label_race']

    def process_batch(self, batch):
        img, lab_disease, lab_sex, lab_race = self.unpack_batch(batch)
        out_disease, out_sex, out_race = self.forward(img)
        loss_disease = F.binary_cross_entropy(torch.sigmoid(out_disease), lab_disease)
        loss_sex = F.cross_entropy(out_sex, lab_sex)
        loss_race = F.cross_entropy(out_race, lab_race, weight=self.class_weights_race.type_as(img))

        if self.confusion == 'race-confusion':
            loss_confusion = -torch.mean(torch.log_softmax(out_race, dim=1))
        elif self.confusion == 'sex-confusion':
            loss_confusion = -torch.mean(torch.log_softmax(out_sex, dim=1))
        else: 
            loss_confusion = 0

        return loss_disease, loss_sex, loss_race, loss_confusion

    def training_step(self, batch, batch_idx, optimizer_idx):
        omega = self.alpha / (1 + np.exp(-(self.global_step - self.fading_in_steps) / self.fading_in_range))
        loss_disease, loss_sex, loss_race, loss_confusion = self.process_batch(batch)
        self.log_dict({"train_loss_disease": loss_disease, "train_loss_sex": loss_sex, "train_loss_race": loss_race, "train_loss_confusion": loss_confusion, "omega": omega})
        # grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        # self.logger.experiment.add_image('images', grid, self.global_step)
        
        if optimizer_idx == 0:
            return loss_disease #+ omega * loss_confusion
        if optimizer_idx == 1:
            return loss_sex
        if optimizer_idx == 2:
            return loss_race
        if optimizer_idx == 3:
            if self.confusion is None:
                return None
            if self.confusion == 'race-confusion' or self.confusion == 'sex-confusion':
                return omega * loss_confusion
            if self.confusion == 'race-negation':
                return - omega * loss_race 
            if self.confusion == 'sex-negation':
                return - omega * loss_sex 
            

                # V1: negate race classification lose
            #return  # V1: negate sex classification lose
            # return omega * loss_confusion # V2: confusion loss

    def validation_step(self, batch, batch_idx):
        loss_disease, loss_sex, loss_race, loss_confusion = self.process_batch(batch)
        self.validation_step_outputs.append(loss_disease)
        loss_disease_mod = loss_disease if self.global_step > self.fading_in_steps else 1.0
        self.log_dict({"val_loss_disease_mod": loss_disease_mod, "val_loss_disease": loss_disease, "val_loss_sex": loss_sex, "val_loss_race": loss_race, "val_loss_confusion": loss_confusion})
    
    def on_validation_epoch_end(self):
        loss = torch.mean(torch.stack(self.validation_step_outputs))
        session.report({"loss": loss})
        self.log('val_loss_epoch_end', loss, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss_disease, loss_sex, loss_race, loss_confusion = self.process_batch(batch)
        self.log_dict({"test_loss_disease": loss_disease, "test_loss_sex": loss_sex, "test_loss_race": loss_race, "test_loss_confusion": loss_confusion})

"""

"""
class DenseNetXRV(pl.LightningModule):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        if self.model_name == 'imagenet':
            self.backbone = models.densenet121(pretrained=True)
        else:
            self.model = xrv.models.DenseNet(weights=f"densenet121-res224-{self.model_name}")
            self.model.op_threshs = None
        num_features = self.model.classifier.in_features
        #TOD freeze(self.model)
        self.model.classifier = nn.Linear(num_features, self.num_classes)

    def remove_head(self):
        num_features = self.model.classifier.in_features
        id_layer = nn.Identity(num_features)
        self.model.classifier = id_layer

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=0.001)
        return optimizer

    def unpack_batch(self, batch):
        return batch['image'], batch['label']

    def process_batch(self, batch):
        img, lab = self.unpack_batch(batch)
        out = self.forward(img)
        prob = torch.sigmoid(out)
        loss = F.binary_cross_entropy(prob, lab)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss)
        grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        self.logger.experiment.add_image('images', grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('test_loss', loss)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True