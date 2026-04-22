import os
import torch
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

CURRENT_DIR = os.getcwd()
DATA_DIR = CURRENT_DIR + "/data/PCXP"
TEST_DIR = DATA_DIR + "/test"
TRAIN_DIR = DATA_DIR + "/train"
MODEL_DIR = CURRENT_DIR + "/models"

TARGET_SIZE = 224
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.CenterCrop(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=transform)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = train_dataset.classes
num_classes = len(class_names)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)


model.to(device)


if os.path.exists(MODEL_DIR + "/model.pth"):
    print("Model found, loading...")
    model.load_state_dict(torch.load(MODEL_DIR + "/model.pth", map_location=device))


model.eval()
correct = 0
total = 0


counter_list = []
for label in os.listdir(TEST_DIR):
    counter = 0
    for images in os.listdir(TEST_DIR + "/" + label):
        counter += 1
    counter_list.append((label, counter))
sum_images = counter_list[0][1] + counter_list[1][1]
counter_list.append(("Total", sum_images))
print(counter_list)

all_preds = []
all_labels = []

model.eval()

threshold = 0.87
with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        probs = F.softmax(outputs, dim=1)

        pneumonia_probs = probs[:, 1]
        preds = (pneumonia_probs >= threshold).long() # If prob > threshold, then 1 (Pneumonia), else 0 (Normal)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())



cm = confusion_matrix(all_labels, all_preds)
recall = cm[1][1] / (cm[1][0] + cm[1][1])
print(f"Using threshold: {threshold}")
print("Confusion matrix:")
print(cm)
print(f"Pneumonia recall: {recall:.4f}")

