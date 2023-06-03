#!/usr/bin/env python

"""clip_classes.py: Contains the CLIPDataset and CLIPTrainer wrapper"""

__author__ = "Andrew D'Amico, Christoper Alexander, Katya Nosulko, Vivek Chamala, Matthew Conger"
__copyright__ = "Copyright 2023"
__credits__ = ["Rob Knight", "Peter Maxwell", "Gavin Huttley",
                    "Matthew Wakefield"]
__license__ = ""
__version__ = "0.0.1"
__maintainer__ = "Andrew Damico"
__email__ = "andrew.damico@u.northwestern.edu"


from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
import torch
from torch.cuda.amp import autocast

def get_transforms(mode='train'):
    
    '''
    Performs image augmentation
    '''
    
    if mode == 'train':
        transforms = torch.nn.Sequential(
            [
                #transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAutocontrast(),
                transforms.RandomAdjustSharpness(sharpness_factor=2),
                transforms.RandomRotation(degrees=(15)),
                #transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD)
            ]
        )
        
    else:
        transforms = torch.nn.Sequential(
            [
                #transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                #transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD)
            ]
        )
        
    return torch.jit.script(transforms)
    


class CLIPDataset(Dataset):
    def __init__(self, 
                 image_paths: list, 
                 text: list, 
                 #mode: str = 'train'
                 transformations
                ):
        
        self.image_paths = image_paths
        self.captions = text #already a list?
        self.tokens = tokenizer(
            self.captions, 
            padding = 'max_length',
            truncation = True,
            max_length = MAX_LEN
        )
        
        self.transforms = transformations #new

    def __getitem__(self, idx):
        token = self.tokens[idx]
        
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.tokens.items()
        }
        
        image = open(self.image_paths[idx])
        image = image.convert('RGB')
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]
        
        return item
        
        #return {
        #    'input_ids': token.ids, 
        #    'attention_mask': token.attention_mask,
        #    'pixel_values': self.augment(Image.open(self.image_paths[idx]).convert('RGB'))}

    def __len__(self):
        #return len(self.image_paths)
        return len(self.captions)
    
class CLIPTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs, return_loss=True)
        return outputs["loss"]

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            try: #Bypassed the use_amp since this was generating an error.
                if self.use_amp:
                    with autocast():
                        loss = self.compute_loss(model, inputs)
                else:
                    loss = self.compute_loss(model, inputs)
            except:
                loss = self.compute_loss(model, inputs)
        
        #self.log("val_loss", loss)
        
        return (loss, None, None)

