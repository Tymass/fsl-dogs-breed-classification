import torch
import torch.utils.data as data
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import PrototypicalNetwork, map_classes, get_test_transform, ImageDataset, rename_class, help_info
import argparse

# funkcja do wyswietlania nazw klas w zbiorze testowym
def print_test_classes(class_dict, labels):
    found_classes = []
    for label in np.unique(labels):
        for class_name, class_num in class_dict.items():
            if class_num == label:
                found_classes.append(rename_class(class_name))
                break

    classes_str = ", ".join(found_classes)
    print(f"Testowe klasy: {classes_str}")

# funkcja do wyswietlania macierzy pomylek
def plot_conf_mat(labels_np=None, predicted_labels=None):
    conf_mat = confusion_matrix(labels_np, predicted_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='g', cmap='Blues')
    plt.ylabel('Przewidziane Klasy')
    plt.xlabel('Rzeczywiste Klasy')
    plt.title('Macierz Pomyłek')
    #plt.savefig('conf-mat-6.png')
    plt.show()


@torch.no_grad()
def test_proto_net(model, dataset, k_shot=4):
    model = model.to(device)                            # laczymy model do ukladu GPU
    model.eval()                                        # wylaczamy tryb treningowy
    total_classes = dataset.get_labels_as_tensor().unique().shape[0]                # liczba klas 
    samples_per_class = dataset.get_labels_as_tensor().shape[0] // total_classes    # liczba przykladow dla kazdej klasy w zbiorze testowym

    loader = data.DataLoader(dataset, batch_size=128, num_workers=0, shuffle=False, drop_last=False)    # dataloader do zbioru treningowego

    features_collected = []
    targets_collected = []
    for batch_imgs, batch_targets in tqdm(loader, "Przetwarzanie danych", leave=False):
        batch_imgs = batch_imgs.to(device)
        extracted_features = model.model(batch_imgs)                        # kodujemy cechy zdjec
        features_collected.append(extracted_features.detach().cpu())        # przechowywujemy cechy po odlaczeniu ich od ukladu GPU 
        targets_collected.append(batch_targets)                             # przechowywujemy etykiety
    features_collected = torch.cat(features_collected, dim=0)
    targets_collected = torch.cat(targets_collected, dim=0)

    # sortujemy i zmieniamy ksztalt cech i etykiet
    sorted_targets, sorting_indices = targets_collected.sort()
    sorted_targets = sorted_targets.reshape(total_classes, samples_per_class).transpose(0, 1)
    features_sorted = features_collected[sorting_indices].reshape(total_classes, samples_per_class, -1).transpose(0, 1)
    
    for shot_index in tqdm(range(0, features_sorted.shape[0], k_shot), "Przetwarzanie danych", leave=False):
        # tworzymy prototypy dla K elementow
        shot_features, shot_targets = features_sorted[shot_index:shot_index+k_shot].flatten(0,1), sorted_targets[shot_index:shot_index+k_shot].flatten(0,1)
        class_centroids, centroid_classes = model.calculate_prototypes(shot_features, shot_targets)
        # weryfikujemy dzialanie modelu na pozostalych elementach
        for eval_index in range(0, features_sorted.shape[0], k_shot):
            if shot_index == eval_index:  
                continue
            eval_features, eval_targets = features_sorted[eval_index:eval_index+k_shot].flatten(0,1), sorted_targets[eval_index:eval_index+k_shot].flatten(0,1)
            predictions, matched_labels, _ = model.classify_feats(class_centroids, centroid_classes, eval_features, eval_targets)
            
    return (predictions, matched_labels), (class_centroids, centroid_classes)

# funkcja do wyswietlania wartosci metryk 
def classif_report(num_classes=12, labels_np=None, predicted_labels=None):
    report = classification_report(labels_np, predicted_labels, target_names=[f'Klasa {i}' for i in range(num_classes)])
    print(report)
    print(f'Dokladnosc modelu: {accuracy_score(labels_np, predicted_labels)}')


parser = argparse.ArgumentParser(description='Skrypt do testow modelu predykcyjnego')

parser.add_argument('--path', type=str, help=help_info['path'])
parser.add_argument('--conf', type=bool, default=False, help=help_info['conf'])
parser.add_argument('--report', type=bool, default=False, help=help_info['report'])
parser.add_argument('-K', type=int, default=5, help=help_info['K'])

args = parser.parse_args()

conf = args.conf
report = args.report
K= args.K
model_path = args.path


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

test_set = ImageDataset("Dataset/test", label_map=map_classes(), img_transform=get_test_transform())

protonet_model = PrototypicalNetwork.load_from_checkpoint(model_path)       # ladujemy model po sciezce 
protonet_model.eval()                                                       # 

resoults, class_prototypes = test_proto_net(protonet_model, test_set, k_shot=K) # testujemy model

preds_np = resoults[0].numpy()
predicted_labels = np.argmax(preds_np, axis=1)
labels_np = resoults[1].numpy()


if report:
    classif_report(labels_np=labels_np, predicted_labels=predicted_labels)
if conf:
    plot_conf_mat(labels_np=labels_np, predicted_labels=predicted_labels)





