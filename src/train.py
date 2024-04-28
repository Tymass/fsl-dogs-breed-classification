import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from utils import BatchSampler, ImageDataset, split_batch, PrototypicalNetwork, map_classes, get_test_transform, get_train_transform, help_info

# funkcja do wyswietlania przykladow z subsetu danych
def display_data_examples(examples_num=10, data_subset=None):
    stanford_images = torch.stack([data_subset[np.random.randint(len(data_subset))][0] for idx in range(examples_num)], dim=0)
    img_grid = torchvision.utils.make_grid(stanford_images, nrow=6, normalize=True, pad_value=0.9)
    img_grid = img_grid.permute(1, 2, 0)
    plt.figure(figsize=(8,8))
    plt.title("Przykładowe zdjęcia ze zbioru danych Stanford Dogs")
    plt.imshow(img_grid)
    plt.axis('off')
    plt.show()
    plt.close()

# funkcja do wyswietlania danych w pojedynczej iteracji dataloadera
def show_example_batch(data_loader):
    imgs, targets = next(iter(data_loader))                                                     # dokonujemy pojedynczej iteracji
    support_imgs, query_imgs, _, _ = split_batch(imgs, targets)                                 # rozdzielamy subsety
    support_grid = torchvision.utils.make_grid(support_imgs, nrow=K_SHOT, normalize=True, pad_value=0.9) # tworzymy siatke zdjec do supportsetu
    support_grid = support_grid.permute(1, 2, 0)                                                         # przeksztalcamy wymiar danych (potrzebne do wyswietlania)
    query_grid = torchvision.utils.make_grid(query_imgs, nrow=K_SHOT, normalize=True, pad_value=0.9)
    query_grid = query_grid.permute(1, 2, 0)

    fig, ax = plt.subplots(1, 2, figsize=(8, 5))             # tworzymy wykres dla obu subsetow
    ax[0].imshow(support_grid)
    ax[0].set_title("Zbiór pomocniczy")
    ax[0].axis('off')
    ax[1].imshow(query_grid)
    ax[1].set_title("Zbiór zapytań")
    ax[1].axis('off')
    plt.suptitle("Zadanie FSL", weight='bold')
    #plt.savefig("fsl-task.png")
    plt.show()
    plt.close()

# funkcja do trenowania modelu
def train_model(model_class, train_loader, val_loader, epochs, **kwargs):
    # konfigurujemy trenera
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, model_class.__name__),  # sciezka do zapisu modelu
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",        # uklad GPU jesli dostepny
                         devices=1,
                         max_epochs=epochs,                                                     # liczba epok treningowcyh
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch")],
                         enable_progress_bar=True)
    trainer.logger._default_hp_metric = None

    pl.seed_everything(42)  # ustawiamy seed w celu zapewnienia reprodukowywalnosci 
    model = model_class(**kwargs)
    trainer.fit(model, train_loader, val_loader)
    model = model_class.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)  # na koncu treningu ladujemy najlepszy checkpoint

    return model


print("Inicjalizacja środowiska")

# inicjalizujemy parsera danych do skryptu
parser = argparse.ArgumentParser(description='Skrypt do treningu modelu predykcyjnego')

parser.add_argument('--l_r', type=float, default=0.001, help=help_info['learning_rate'])
parser.add_argument('--epochs', type=int, default=1, help=help_info['epochs'])
parser.add_argument('-K', type=int, default=5, help=help_info['K'])
parser.add_argument('-M', type=int, default=3, help=help_info['M'])
parser.add_argument('--dim', type=int, default=64, help=help_info['embedded_dim'])

args = parser.parse_args()


CHECKPOINT_PATH = os.getcwd() + '/models'
N_WAY = args.M
K_SHOT = args.K
LR = args.l_r
EMBEDDED_DIM = args.dim
EPOCHS = args.epochs

print("Wspolczynnik uczenia: ", args.l_r)
print("Liczba epok: ", args.epochs)
print("K: ", args.K)
print("M: ", args.M)
print("Wymiar przestrzeni ukrytej: ", args.dim)

pl.seed_everything(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("Szukanie dostepnego ukladu GPU")
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu") # podlaczamy uklad GPU
print("Wykryte urzadzenie:", device)


torch.manual_seed(0)           
classes = torch.randperm(120)                                                               # losujemy klasy (ten sam seed - te same klasy co w innych modulach)
train_classes, val_classes, test_classes = classes[:96], classes[96:108], classes[108:]


test_transform = get_test_transform()           # pobieramy tranformacje dancch
train_transform = get_train_transform()


train_set = ImageDataset("Dataset/train", label_map=map_classes(), img_transform=train_transform)   # tworzymy zbiory danych
val_set = ImageDataset("Dataset/val", label_map=map_classes(), img_transform=test_transform)

                                                                                                    # tworzymy dataloadery
train_data_loader = data.DataLoader(train_set,
                                    batch_sampler=BatchSampler(group_targets=train_set.get_labels_as_tensor(),
                                                                      double_support=True,
                                                                      Group_count=N_WAY,
                                                                      Sample_count=K_SHOT,
                                                                      randomize=True),
                                    num_workers=2)

val_data_loader = data.DataLoader(val_set,
                                  batch_sampler=BatchSampler(group_targets=val_set.get_labels_as_tensor(),
                                                                    double_support=True,
                                                                    Group_count=N_WAY,
                                                                    Sample_count=K_SHOT,
                                                                    randomize=False),
                                  num_workers=2)

# wyswietlamy przykladowe dane w zadaniu meta uczenia
show_example_batch(train_data_loader)

print("Rozpoczynanie treningu")

# trenujemy model
protonet_model = train_model(PrototypicalNetwork,
                             train_loader=train_data_loader,
                             val_loader=val_data_loader,
                             epochs=EPOCHS,
                             feature_dim=EMBEDDED_DIM,
                             learning_rate=LR,)



