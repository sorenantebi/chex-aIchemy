import torch
from tqdm import tqdm

"""

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

        logits_disease = torch.cat(logits_disease, dim=0)
        preds_disease = torch.cat(preds_disease, dim=0)
        targets_disease = torch.cat(targets_disease, dim=0)

        logits_sex = torch.cat(logits_sex, dim=0)
        preds_sex = torch.cat(preds_sex, dim=0)
        targets_sex = torch.cat(targets_sex, dim=0)

        logits_race = torch.cat(logits_race, dim=0)
        preds_race = torch.cat(preds_race, dim=0)
        targets_race = torch.cat(targets_race, dim=0)

        counts = []
        for i in range(0,num_classes_disease):
            t = targets_disease[:, i] == 1
            c = torch.sum(t)
            counts.append(c)
        print(counts)

        counts = []
        for i in range(0,num_classes_sex):
            t = targets_sex == i
            c = torch.sum(t)
            counts.append(c)
        print(counts)

        counts = []
        for i in range(0,num_classes_race):
            t = targets_race == i
            c = torch.sum(t)
            counts.append(c)
        print(counts)

    return preds_disease.cpu().numpy(), targets_disease.cpu().numpy(), logits_disease.cpu().numpy(), preds_sex.cpu().numpy(), targets_sex.cpu().numpy(), logits_sex.cpu().numpy(), preds_race.cpu().numpy(), targets_race.cpu().numpy(), logits_race.cpu().numpy()

"""

"""
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
