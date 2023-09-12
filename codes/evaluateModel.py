import argparse
from getData import DDIData
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.metrics import (f1_score, balanced_accuracy_score, 
    classification_report, confusion_matrix, roc_curve, auc)
import torch
import tqdm
from torchvision import transforms as T
import torchvision



def eval_model(model, dataset, use_gpu=False, show_plot=False):
    """Evaluate HAM10000. Assumes the data is split into binary/malignant labels, as this is 
    what our models are trained+evaluated on."""

    use_gpu = (use_gpu and torch.cuda.is_available())
    device = torch.device("cuda") if use_gpu else torch.device("cpu")

    # load dataset
    dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=32, shuffle=False,
                    num_workers=0, pin_memory=use_gpu)

    # prepare model for evaluation
    model.to(device).eval()

    # log output for all images in dataset
    hat, star, all_paths = [], [], []
    for batch in tqdm.tqdm(enumerate(dataloader)):
        i, (paths, images, target, skin_tone) = batch
        images = images.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(images)

        hat.append(output[:,1].detach().cpu().numpy())
        star.append(target.cpu().numpy())
        all_paths.append(paths)

    hat = np.concatenate(hat)
    star = np.concatenate(star)
    all_paths = np.concatenate(all_paths)
    threshold = model._ddi_threshold
    m_name = model._ddi_name
    m_web_path = model._ddi_web_path

    report = classification_report(star, (hat>threshold).astype(int), 
        target_names=["benign","malignant"])
    fpr, tpr, _ = roc_curve(star, hat, pos_label=1,
                                sample_weight=None,
                                drop_intermediate=True)
    auc_est = auc(fpr, tpr)

    if show_plot:
        _=plt.plot(fpr, tpr, 
            color="blue", linestyle="-", linewidth=2, 
            marker="o", markersize=2, 
            label=f"AUC={auc_est:.3f}")[0]
        plt.show()
        plt.close()

    eval_results = {'predicted_labels':hat, # predicted labels by model
                    'true_labels':star,     # true labels
                    'images':all_paths,     # image paths
                    'report':report,        # sklearn classification report
                    'ROC_AUC':auc_est,      # ROC-AUC
                    'threshold':threshold,  # >= threshold ==> malignant
                    'model':m_name,         # model name
                    'web_path':m_web_path,  # web link to download model
                    }

    return eval_results




if __name__ == '__main__':
    # get arguments from command line
    # load model 
    model_path = os.path.join("DDI-models", "HAM10000.pth")
    model = torchvision.models.inception_v3(pretrained=False, transform_input=True)
    model.fc = torch.nn.Linear(2048, 2)
    model.AuxLogits.fc = torch.nn.Linear(768, 2)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model._ddi_name = 'HAM10000'
    model._ddi_threshold = 0.733
    means = [0.485, 0.456, 0.406]
    stds  = [0.229, 0.224, 0.225]
    test_transform = T.Compose([
        lambda x: x.convert('RGB'),
        T.Resize(299),
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(mean=means, std=stds)
    ])
    # load DDI dataset
    root = os.path.dirname(os.path.dirname(__file__))
    dataset = DDIData(root,transform=test_transform)
    # evaluate results on data
    eval_results = eval_model(model, dataset, 
        use_gpu=False, show_plot=True)
    root = os.path.dirname(os.path.dirname(__file__))
    eval_save_path=os.path.join(root, 'results','HAM10000.pkl')
        # save evaluation results in a pickle file 
    
    with open(eval_save_path, 'wb') as f:
        pickle.dump(eval_results, f)
