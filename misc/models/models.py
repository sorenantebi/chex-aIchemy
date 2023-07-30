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

class MultitaskHead(nn.Module):
    def __init__(self, num_features, num_classes_disease: int = 0, num_classes_sex: int = 0, num_classes_race: int = 0):
        super().__init__()
        self.fc_disease = nn.Linear(num_features, num_classes_disease)
        self.fc_sex = nn.Linear(num_features, num_classes_sex)
        self.fc_race = nn.Linear(num_features, num_classes_race)

    def forward(self, embedding):
        out_disease = self.fc_disease(embedding)
        out_sex = self.fc_sex(embedding)
        out_race = self.fc_race(embedding)
        return out_disease, out_sex, out_race
"""

"""
class DenseNetMultitask(pl.LightningModule):
    def __init__(self, args: Hparams):
        super().__init__()
        self.automatic_optimization = False
        self.alpha = args.alpha
        self.fading_in_steps = args.fading_in_steps
        self.fading_in_range = args.fading_in_range
        self.confusion = args.confusion
        self.validation_step_outputs = []
        self.label_noise = True if args.label_noise == 'True' else False
        self.lr_d = args.lr_d
        self.lr_s = args.lr_s
        self.lr_r = args.lr_r
        self.lr_b = args.lr_b

        self.class_weights_race = torch.FloatTensor(tuple(args.class_weights_race))

        if args.model_name == 'imagenet':
            self.backbone = models.densenet121(pretrained=True)
        else:
            self.backbone = xrv.models.DenseNet(weights=f"densenet121-res224-{args.model_name}")
            self.backbone.op_threshs = None

        num_features = self.backbone.classifier.in_features
        self.classification_head = MultitaskHead(num_features,args.num_classes_disease,args.num_classes_sex,args.num_classes_race)
        self.fc_connect = nn.Identity(num_features)
        self.backbone.classifier = self.fc_connect

    def forward(self, x):
        embedding = self.backbone.forward(x)
        return self.classification_head(embedding)
    
    def initialize_parameters(self):
        params_backbone = list(self.backbone.parameters())
        params_disease = params_backbone + list(self.classification_head.fc_disease.parameters())
        
        if self.confusion is None and self.label_noise == False:
            params_sex = params_backbone + list(self.classification_head.fc_sex.parameters())
            params_race = params_backbone + list(self.classification_head.fc_race.parameters())
        else: 
            params_sex = list(self.classification_head.fc_sex.parameters())
            params_race = list(self.classification_head.fc_race.parameters()) 

        return params_disease, params_sex, params_race, params_backbone
    
    def configure_optimizers(self):
        params_disease, params_sex, params_race, params_backbone = self.initialize_parameters()
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
        loss_sex= F.cross_entropy(out_sex, lab_sex)
        loss_race =F.cross_entropy(out_race, lab_race, weight=self.class_weights_race.type_as(img))

        #calculate confusion loss
        _dict = {
            'race-confusion' : -torch.mean(torch.log_softmax(out_race, dim=1)),
            'sex-confusion' : -torch.mean(torch.log_softmax(out_sex, dim=1))
        }
        loss_confusion = _dict.get(self.confusion, 0)
        
       
        return loss_disease, loss_sex, loss_race, loss_confusion

    def optimize(self, optimizer, batch, idx):
        omega = self.alpha / (1 + np.exp(-(self.global_step - self.fading_in_steps) / self.fading_in_range))
        loss_disease, loss_sex, loss_race, loss_confusion = self.process_batch(batch)
        self.log_dict({"train_loss_disease": loss_disease, "train_loss_sex": loss_sex, "train_loss_race": loss_race, "train_loss_confusion": loss_confusion, "omega": omega})
        
        """   _dict = {
            0: lambda: self.opt_step(optimizer=optimizer, loss=loss_disease),
            1: lambda: self.opt_step(optimizer=optimizer, loss=loss_sex),
            2: lambda: self.opt_step(optimizer=optimizer, loss=loss_race),
            3: {None: lambda: self.untoggle_optimizer(optimizer),
                'race-confusion': lambda: self.opt_step(optimizer=optimizer, loss= omega * loss_confusion), 
                'sex-confusion': lambda: self.opt_step(optimizer=optimizer, loss= omega * loss_confusion),
                'race-negation': lambda: self.opt_step(optimizer=optimizer, loss= - omega * loss_race),
                 'sex-negation': lambda: self.opt_step(optimizer=optimizer, loss= - omega * loss_sex) }.get(self.confusion)
        }
        selected_function = _dict.get(idx) # maybe add error message
        selected_function()
        """

        if idx == 0:
            self.opt_step(optimizer=optimizer, loss=loss_disease)
        if idx == 1:
            self.opt_step(optimizer=optimizer, loss=loss_sex)
        if idx == 2:
            self.opt_step(optimizer=optimizer, loss=loss_race)
        if idx == 3:
            if self.confusion is None:
                self.untoggle_optimizer(optimizer)
            elif self.confusion in {'race-confusion', 'sex-confusion'}:
                self.opt_step(optimizer=optimizer, loss=omega * loss_confusion)
            elif self.confusion == 'race-negation':
                self.opt_step(optimizer=optimizer, loss= - omega * loss_race)
            elif self.confusion == 'sex-negation':
                self.opt_step(optimizer=optimizer, loss= - omega * loss_sex)
            else:
                raise ValueError("Invalid adversary")
   
    def opt_step(self, optimizer, loss):
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

    def training_step(self, batch, batch_idx):
        optimizer_list = [*self.optimizers()]
        for idx, optimizer in enumerate(optimizer_list):
            self.optimize(optimizer=optimizer, batch=batch, idx=idx)
     
    def validation_step(self, batch, batch_idx):
        loss_disease, loss_sex, loss_race, loss_confusion = self.process_batch(batch)
        self.validation_step_outputs.append(loss_disease)
        loss_disease_mod = loss_disease if self.global_step > self.fading_in_steps else 1.0
        self.log_dict({"val_loss_disease_mod": loss_disease_mod, "val_loss_disease": loss_disease, "val_loss_sex": loss_sex, "val_loss_race": loss_race, "val_loss_confusion": loss_confusion})
    
    def on_validation_epoch_end(self):
        loss = torch.mean(torch.stack(self.validation_step_outputs))
        self.log('loss', loss, on_epoch=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        loss_disease, loss_sex, loss_race, loss_confusion = self.process_batch(batch)
        self.log_dict({"test_loss_disease": loss_disease, "test_loss_sex": loss_sex, "test_loss_race": loss_race, "test_loss_confusion": loss_confusion})

"""

"""
class DenseNet(pl.LightningModule):
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
        params_to_update = [
            param for param in self.parameters() if param.requires_grad == True
        ]
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