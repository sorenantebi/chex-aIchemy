import torch
from tqdm import tqdm
import os
import pandas as pd

"""
For Multitask and Adversarial training
"""
def test_multi(model, data_loader, device, num_classes_disease, num_classes_sex, num_classes_race):
    model.eval()
    logits_disease = []
    preds_disease = []
    targets_disease = []
    logits_sex = []
    preds_sex = []
    targets_sex = []
    logits_race = []
    preds_race = []
    targets_race = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab_disease, lab_sex, lab_race = batch['image'].to(device), batch['label_disease'].to(device), batch['label_sex'].to(device), batch['label_race'].to(device)
            out_disease, out_sex, out_race = model(img)

            pred_disease = torch.sigmoid(out_disease)
            pred_sex = torch.softmax(out_sex, dim=1)
            pred_race = torch.softmax(out_race, dim=1)

            logits_disease.append(out_disease)
            preds_disease.append(pred_disease)
            targets_disease.append(lab_disease)

            logits_sex.append(out_sex)
            preds_sex.append(pred_sex)
            targets_sex.append(lab_sex)

            logits_race.append(out_race)
            preds_race.append(pred_race)
            targets_race.append(lab_race)

        logits_disease, preds_disease, targets_disease = concatenate(logits=logits_disease, preds=preds_disease, targets=targets_disease)
        logits_sex, preds_sex, targets_sex = concatenate(logits=logits_sex, preds=preds_sex, targets=targets_sex)
        logits_race, preds_race, targets_race = concatenate(logits=logits_race, preds=preds_race, targets=targets_race)
      
    
        counts = []
        for i in range(0,num_classes_disease):
            t = targets_disease[:, i] == 1
            c = torch.sum(t)
            counts.append(c)
        print(counts)

        print_counts(num_classes_sex, targets_sex)
        print_counts(num_classes_race, targets_race)

    return preds_disease.cpu().numpy(), targets_disease.cpu().numpy(), logits_disease.cpu().numpy(), preds_sex.cpu().numpy(), targets_sex.cpu().numpy(), logits_sex.cpu().numpy(), preds_race.cpu().numpy(), targets_race.cpu().numpy(), logits_race.cpu().numpy()

def concatenate(logits, preds, targets):
    return torch.cat(logits, dim=0), torch.cat(preds, dim=0), torch.cat(targets, dim=0)

     
def print_counts(num_classes, targets):
    counts = []
    for i in range(num_classes):
        t = targets == i
        c = torch.sum(t)
        counts.append(c)
    print(counts)

def embeddings(model, data_loader, device):
    model.eval()

    embeds = []
    targets_disease = []
    targets_sex = []
    targets_race = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab_disease, lab_sex, lab_race = batch['image'].to(device), batch['label_disease'].to(device), batch['label_sex'].to(device), batch['label_race'].to(device)
            emb = model.backbone(img)
            embeds.append(emb)
            targets_disease.append(lab_disease)
            targets_sex.append(lab_sex)
            targets_race.append(lab_race)

        embeds = torch.cat(embeds, dim=0)
        targets_disease = torch.cat(targets_disease, dim=0)
        targets_sex = torch.cat(targets_sex, dim=0)
        targets_race = torch.cat(targets_race, dim=0)

    return embeds.cpu().numpy(), targets_disease.cpu().numpy(), targets_sex.cpu().numpy(), targets_race.cpu().numpy()

def analysis(out_dir, num_classes_disease, num_classes_sex, num_classes_race, model, data, device):
    
    cols_names_classes_disease = [
        f'class_{str(i)}' for i in range(num_classes_disease)
    ]

    cols_names_logits_disease = [
        f'logit_{str(i)}' for i in range(num_classes_disease)
    ]
    
    cols_names_targets_disease = [
        f'target_{str(i)}' for i in range(num_classes_disease)
    ]

    cols_names_classes_sex = [f'class_{str(i)}' for i in range(num_classes_sex)]
    cols_names_logits_sex = [f'logit_{str(i)}' for i in range(num_classes_sex)]

    cols_names_classes_race = [f'class_{str(i)}' for i in range(num_classes_race)]
    cols_names_logits_race = [f'logit_{str(i)}' for i in range(num_classes_race)]

    print('VALIDATION')
    preds_val_disease, targets_val_disease, logits_val_disease, preds_val_sex, targets_val_sex, logits_val_sex, preds_val_race, targets_val_race, logits_val_race = test_multi(model, data.val_dataloader(), device,  num_classes_disease, num_classes_sex, num_classes_race)

    df = pd.DataFrame(data=preds_val_disease, columns=cols_names_classes_disease)
    df_logits = pd.DataFrame(data=logits_val_disease, columns=cols_names_logits_disease)
    df_targets = pd.DataFrame(data=targets_val_disease, columns=cols_names_targets_disease)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions_val_disease.csv'), index=False)

    df = pd.DataFrame(data=preds_val_sex, columns=cols_names_classes_sex)
    df_logits = pd.DataFrame(data=logits_val_sex, columns=cols_names_logits_sex)
    df = pd.concat([df, df_logits], axis=1)
    df['target'] = targets_val_sex
    df.to_csv(os.path.join(out_dir, 'predictions_val_sex.csv'), index=False)

    df = pd.DataFrame(data=preds_val_race, columns=cols_names_classes_race)
    df_logits = pd.DataFrame(data=logits_val_race, columns=cols_names_logits_race)
    df = pd.concat([df, df_logits], axis=1)
    df['target'] = targets_val_race
    df.to_csv(os.path.join(out_dir, 'predictions_val_race.csv'), index=False)

    print('TESTING')
    preds_test_disease, targets_test_disease, logits_test_disease, preds_test_sex, targets_test_sex, logits_test_sex, preds_test_race, targets_test_race, logits_test_race = test_multi(model, data.test_dataloader(), device,  num_classes_disease, num_classes_sex, num_classes_race)

    df = pd.DataFrame(data=preds_test_disease, columns=cols_names_classes_disease)
    df_logits = pd.DataFrame(data=logits_test_disease, columns=cols_names_logits_disease)
    df_targets = pd.DataFrame(data=targets_test_disease, columns=cols_names_targets_disease)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions_test_disease.csv'), index=False)

    df = pd.DataFrame(data=preds_test_sex, columns=cols_names_classes_sex)
    df_logits = pd.DataFrame(data=logits_test_sex, columns=cols_names_logits_sex)
    df = pd.concat([df, df_logits], axis=1)
    df['target'] = targets_test_sex
    df.to_csv(os.path.join(out_dir, 'predictions_test_sex.csv'), index=False)

    df = pd.DataFrame(data=preds_test_race, columns=cols_names_classes_race)
    df_logits = pd.DataFrame(data=logits_test_race, columns=cols_names_logits_race)
    df = pd.concat([df, df_logits], axis=1)
    df['target'] = targets_test_race
    df.to_csv(os.path.join(out_dir, 'predictions_test_race.csv'), index=False)

    print('EMBEDDINGS')

    embeds_val, targets_val_disease, targets_val_sex, targets_val_race = embeddings(model, data.val_dataloader(), device)
    df = pd.DataFrame(data=embeds_val)
    df_targets_disease = pd.DataFrame(data=targets_val_disease, columns=cols_names_targets_disease)
    df = pd.concat([df, df_targets_disease], axis=1)
    df['target_sex'] = targets_val_sex
    df['target_race'] = targets_val_race
    df.to_csv(os.path.join(out_dir, 'embeddings_val.csv'), index=False)

    embeds_test, targets_test_disease, targets_test_sex, targets_test_race = embeddings(model, data.test_dataloader(), device)
    df = pd.DataFrame(data=embeds_test)
    df_targets_disease = pd.DataFrame(data=targets_test_disease, columns=cols_names_targets_disease)
    df = pd.concat([df, df_targets_disease], axis=1)
    df['target_sex'] = targets_test_sex
    df['target_race'] = targets_test_race
    df.to_csv(os.path.join(out_dir, 'embeddings_test.csv'), index=False)



def test(model, data_loader, device):
    model.eval()
    logits = []
    preds = []
    targets = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab = batch['image'].to(device), batch['label'].to(device)
            out = model(img)
            pred = torch.sigmoid(out)
            logits.append(out)
            preds.append(pred)
            targets.append(lab)

        logits = torch.cat(logits, dim=0)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        counts = []
        for i in range(0,num_classes):
            t = targets[:, i] == 1
            c = torch.sum(t)
            counts.append(c)
        print(counts)

    return preds.cpu().numpy(), targets.cpu().numpy(), logits.cpu().numpy()


def embeddings(model, data_loader, device):
    model.eval()

    embeds = []
    targets = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab = batch['image'].to(device), batch['label'].to(device)
            emb = model(img)
            embeds.append(emb)
            targets.append(lab)

        embeds = torch.cat(embeds, dim=0)
        targets = torch.cat(targets, dim=0)

    return embeds.cpu().numpy(), targets.cpu().numpy()

""" 
For implementing a single task model

def analysis(num_classes, model, data, device, out_dir)
    cols_names_classes = ['class_' + str(i) for i in range(0,num_classes)]
    cols_names_logits = ['logit_' + str(i) for i in range(0, num_classes)]
    cols_names_targets = ['target_' + str(i) for i in range(0, num_classes)]

    print('VALIDATION')
    preds_val, targets_val, logits_val = test(model, data.val_dataloader(), device)
    df = pd.DataFrame(data=preds_val, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_val, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_val, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions.val.csv'), index=False)

    print('TESTING')
    preds_test, targets_test, logits_test = test(model, data.test_dataloader(), device)
    df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions.test.csv'), index=False)

    print('EMBEDDINGS')

    model.remove_head()

    embeds_val, targets_val = embeddings(model, data.val_dataloader(), device)
    df = pd.DataFrame(data=embeds_val)
    df_targets = pd.DataFrame(data=targets_val, columns=cols_names_targets)
    df = pd.concat([df, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'embeddings.val.csv'), index=False)

    embeds_test, targets_test = embeddings(model, data.test_dataloader(), device)
    df = pd.DataFrame(data=embeds_test)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'embeddings.test.csv'), index=False)

 """