    import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import types

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)
# -------------------------------
# DATASET
# -------------------------------

import kagglehub
path = kagglehub.dataset_download("fanconic/skin-cancer-malignant-vs-benign")

TRAIN_PATH = os.path.join(path,"data","train")
TEST_PATH = os.path.join(path,"data","test")

# -------------------------------
# CONFIG
# -------------------------------

BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
MODEL_PATH = "densenet_skin_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# -------------------------------
# TRANSFORMS
# -------------------------------

train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

dataset = datasets.ImageFolder(TRAIN_PATH)

train_idx,val_idx = train_test_split(
    list(range(len(dataset))),
    test_size=0.2,
    stratify=dataset.targets,
    random_state=42
)

train_ds = Subset(
    datasets.ImageFolder(TRAIN_PATH,transform=train_tf),
    train_idx
)

val_ds = Subset(
    datasets.ImageFolder(TRAIN_PATH,transform=val_tf),
    val_idx
)

train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
val_loader = DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False)

class_names = dataset.classes
print("Classes:",class_names)

# -------------------------------
# MODEL
# -------------------------------

model = models.densenet121(
    weights=models.DenseNet121_Weights.IMAGENET1K_V1
)

def forward_fix(self,x):
    f = self.features(x)
    out = F.relu(f,inplace=False)
    out = F.adaptive_avg_pool2d(out,(1,1))
    out = torch.flatten(out,1)
    out = self.classifier(out)
    return out

model.forward = types.MethodType(forward_fix,model)

# Freeze early layers
for p in model.features.parameters():
    p.requires_grad = False

# Unfreeze last DenseNet block for GradCAM
for p in model.features.denseblock4.parameters():
    p.requires_grad = True

num_features=model.classifier.in_features
model.classifier=nn.Linear(num_features,len(class_names))

model=model.to(device)

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.classifier.parameters(),lr=LR)

# -------------------------------
# GRADCAM
# -------------------------------

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def forward_hook(module, input, output):
            self.activations = output

            if output.requires_grad:
                output.retain_grad()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):

        self.model.eval()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output)

        score = output[:, class_idx]

        self.model.zero_grad()
        score.backward()

        grads = self.gradients
        acts = self.activations

        if grads is None or acts is None:
            raise RuntimeError("GradCAM gradients not captured")

        weights = torch.mean(grads, dim=(2,3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1, keepdim=True)

        cam = torch.sum(weights * acts, dim=1)

        cam = F.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()

        cam = cv2.resize(cam,(224,224))

        cam = (cam - cam.min())/(cam.max()+1e-8)

        return cam

target_layer = model.features.denseblock4
gradcam = GradCAM(model,target_layer)


train_acc_history = []
train_loss_history = []
# -------------------------------
# TRAIN
# -------------------------------
def train():

    for epoch in range(EPOCHS):

        model.train()

        correct = 0
        total = 0
        running_loss = 0

        for x,y in train_loader:

            x=x.to(device)
            y=y.to(device)

            optimizer.zero_grad()

            out=model(x)

            loss=criterion(out,y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            pred=torch.argmax(out,1)

            correct += (pred==y).sum().item()
            total += y.size(0)

        train_acc = correct/total
        avg_loss = running_loss/len(train_loader)

        train_acc_history.append(train_acc)
        train_loss_history.append(avg_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss:{avg_loss:.4f} | TrainAcc:{train_acc:.4f}")

def plot_training_graphs():

    epochs = range(1, len(train_acc_history)+1)

    plt.figure(figsize=(12,5))

    # Accuracy Graph
    plt.subplot(1,2,1)
    plt.plot(epochs, train_acc_history, marker='o')
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    # Loss Graph
    plt.subplot(1,2,2)
    plt.plot(epochs, train_loss_history, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.show()

def evaluate():

    model.eval()

    preds=[]
    trues=[]
    probs=[]

    with torch.no_grad():

        for x,y in val_loader:

            x=x.to(device)
            y=y.to(device)

            out=model(x)

            prob=torch.softmax(out,dim=1)

            p=torch.argmax(prob,1)

            preds.extend(p.cpu().numpy())
            trues.extend(y.cpu().numpy())
            probs.extend(prob[:,1].cpu().numpy())

    # Accuracy
    acc=accuracy_score(trues,preds)
    print("\nValidation Accuracy:",round(acc,4))

    # Precision Recall F1
    precision=precision_score(trues,preds)
    recall=recall_score(trues,preds)
    f1=f1_score(trues,preds)

    print("Precision:",round(precision,4))
    print("Recall:",round(recall,4))
    print("F1 Score:",round(f1,4))

    print("\nClassification Report\n")
    print(classification_report(trues,preds,target_names=class_names))

    # Confusion Matrix
    cm=confusion_matrix(trues,preds)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC Curve
    fpr,tpr,_=roc_curve(trues,probs)
    roc_auc=auc(fpr,tpr)

    plt.figure()

    plt.plot(fpr,tpr,label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')

    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    plt.show()

    # Precision Recall Curve
    precision_vals, recall_vals, _ = precision_recall_curve(trues, probs)

    plt.figure()

    plt.plot(recall_vals, precision_vals)

    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.show()

    print("AUC Score:",round(roc_auc,4))

def plot_confidence_histogram():

    model.eval()

    confidences = []
    correct_conf = []
    wrong_conf = []

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(device)
            y = y.to(device)

            out = model(x)

            prob = torch.softmax(out, dim=1)

            conf, pred = torch.max(prob, dim=1)

            confidences.extend(conf.cpu().numpy())

            for c, p, t in zip(conf, pred, y):

                if p == t:
                    correct_conf.append(c.cpu().item())
                else:
                    wrong_conf.append(c.cpu().item())

    plt.figure(figsize=(7,5))

    plt.hist(correct_conf, bins=20, alpha=0.7, label="Correct Predictions")
    plt.hist(wrong_conf, bins=20, alpha=0.7, label="Wrong Predictions")

    plt.xlabel("Model Confidence")
    plt.ylabel("Frequency")
    plt.title("Model Confidence Histogram")

    plt.legend()
    plt.show()
  
def show_misclassified_images(max_images=6):

    model.eval()

    wrong_images = []
    wrong_preds = []
    wrong_labels = []

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(device)
            y = y.to(device)

            out = model(x)

            pred = torch.argmax(out, 1)

            for i in range(len(pred)):

                if pred[i] != y[i]:

                    wrong_images.append(x[i].cpu())
                    wrong_preds.append(pred[i].cpu().item())
                    wrong_labels.append(y[i].cpu().item())

                if len(wrong_images) >= max_images:
                    break

            if len(wrong_images) >= max_images:
                break

    if len(wrong_images) == 0:
        print("No misclassified images found!")
        return

    plt.figure(figsize=(12,6))

    for i in range(len(wrong_images)):

        img = wrong_images[i].permute(1,2,0).numpy()

        # denormalize
        mean = np.array([0.485,0.456,0.406])
        std = np.array([0.229,0.224,0.225])
        img = std * img + mean
        img = np.clip(img,0,1)

        plt.subplot(2,3,i+1)
        plt.imshow(img)

        pred_label = class_names[wrong_preds[i]]
        true_label = class_names[wrong_labels[i]]

        plt.title(f"Pred: {pred_label}\nTrue: {true_label}")
        plt.axis("off")

    plt.suptitle("Misclassified Images", fontsize=14)
    plt.tight_layout()
    plt.show()
# -------------------------------
# IMAGE PREDICTION
# -------------------------------

def predict_image(img_path):

    model.eval()

    img=cv2.imread(img_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    original=img.copy()

    img=cv2.resize(img,(224,224))

    tf=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    tensor=tf(img).unsqueeze(0).to(device)

    out=model(tensor)

    pred=torch.argmax(out,1).item()

    label=class_names[pred]

    print("Prediction:",label)

    cam=gradcam.generate(tensor,pred)

    heat=cv2.applyColorMap(np.uint8(255*cam),cv2.COLORMAP_JET)
    heat=cv2.cvtColor(heat,cv2.COLOR_BGR2RGB)/255

    img=original/255

    overlay=(heat+img)
    overlay=overlay/overlay.max()

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(cam,cmap="jet")
    plt.title("GradCAM")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(overlay)
    plt.title(label)
    plt.axis("off")

    plt.show()

# -------------------------------
# RANDOM TEST
# -------------------------------

def test_random():

    benign=os.path.join(TEST_PATH,"benign")
    malignant=os.path.join(TEST_PATH,"malignant")

    imgs=[]

    for folder in [benign,malignant]:
        for f in os.listdir(folder):
            imgs.append(os.path.join(folder,f))

    img=random.choice(imgs)

    print("Testing:",img)

    predict_image(img)

# -------------------------------
# MAIN
# -------------------------------

if __name__=="__main__":

    if os.path.exists(MODEL_PATH):

        model.load_state_dict(torch.load(MODEL_PATH))
        print("Loaded existing model")

    else:

        train()
        torch.save(model.state_dict(),MODEL_PATH)

        plot_training_graphs()
evaluate()

plot_confidence_histogram()

show_misclassified_images()

test_random()