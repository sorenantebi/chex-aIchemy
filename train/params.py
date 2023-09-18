import argparse

"""
Code based on 
https://github.com/biomedia-mira/causal-gen/blob/main/src/main.py
https://github.com/biomedia-mira/causal-gen/blob/main/src/hps.py
"""

class Hparams:
    def update_attributes(self, attr_name, value):
        setattr(self, attr_name, value)

txrv = Hparams()
txrv.confusion = None
txrv.model_name = 'imagenet'     
txrv.image_size = (224, 224)
txrv.num_classes_disease = 14
txrv.num_classes_sex = 2
txrv.num_classes_race = 3
txrv.class_weights_race = (1.0, 1.0, 1.0) 
txrv.batch_size = 150
txrv.epochs = 40
txrv.alpha = 0.0001
txrv.num_workers = 4
txrv.fading_in_steps = 0
txrv.fading_in_range = 1
txrv.img_data_dir = '<path-to-data>'
txrv.lr_d = 0.001
txrv.lr_s = 0.001
txrv.lr_r = 0.001
txrv.lr_b = 0.001
txrv.label_noise = False # To customize noise type, edit in CheXDataModule
txrv.multitask = False


def setup_hparams(parser: argparse.ArgumentParser) -> Hparams:
    model_names = ['all', 'chex', 'pc', 'mimic_ch', 'mimic_nb', 'rsna', 'nih','imagenet']
    confusion_options = [None, 'race-confusion', 'sex-confusion', 'race-negation', 'sex-negation']
    hparams = Hparams()
    args = parser.parse_known_args()[0]
    valid_args = set(args.__dict__.keys())

    if args.confusion is not None:
        txrv.update_attributes('fading_in_steps', 20000)
        txrv.update_attributes('fading_in_range', 800)

    hparams_dict = txrv.__dict__
    print(args)
    for k in hparams_dict.keys():
        if k not in valid_args:
            raise ValueError(f"{k} not in default args")
    parser.set_defaults(**hparams_dict)
 
    for key, value in parser.parse_known_args()[0].__dict__.items():
        hparams.update_attributes(key, value) 
    
    if getattr(hparams, 'model_name') not in model_names:
        raise ValueError(f"{getattr(hparams, 'model_name')} not in model names")
    if getattr(hparams, 'confusion') not in confusion_options:
        raise ValueError(f"{getattr(hparams, 'confusion')} not in confusion options")
    
    print(hparams.__dict__.items())
    return hparams

def add_arguments (parser: argparse.ArgumentParser):
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--image_size', nargs=2, type=int)
    parser.add_argument('--num_classes_disease', type=int)
    parser.add_argument('--num_classes_sex', type=int)
    parser.add_argument('--num_classes_race', type=int)
    parser.add_argument('--class_weights_race', nargs=3, type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--fading_in_steps', type=int) 
    parser.add_argument('--fading_in_range', type=int)
    parser.add_argument('--img_data_dir', type=str)
    parser.add_argument('--lr_d', type=float)
    parser.add_argument('--lr_s', type=float)
    parser.add_argument('--lr_r', type=float)
    parser.add_argument('--lr_b', type=float)
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--dev', default=0)
    parser.add_argument('--confusion')
    parser.add_argument('--label_noise')
    parser.add_argument('--multitask')
    return parser