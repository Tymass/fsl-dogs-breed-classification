import os
from PIL import Image, ImageTk
import random
import torch
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
import tkinter as tk

DATA_MEANS = torch.Tensor([0.52 , 0.48, 0.43])
DATA_STD = torch.Tensor([0.27, 0.24, 0.28])

# slownik flag do skryptow
help_info = {
    'epochs' : 'Liczba epok treningowych. Domyslnie wartosc: 1',
    'learning_rate' : 'Parametr wspolczynnika uczenia. Domyslnie wartosc: 0.001',
    'M' : 'Wspolczynnik M - liczba klas w kazdym zadaniu. Domyslnie wartosc: 5',
    'K' : 'Wspolczynnik K - liczba przykladow na klase. Domyslnie wartosc: 3',
    'embedded_dim' : 'Wymiar przestrzeni ukrytej. Domyslnie wartosc: 64',
    'path' : 'Sciezka do wytrenowanego modelu (rozszezenie .ckpt)',
    'conf' : 'Opcja wyswietlenia macierzy pomylek. Domyslnie wartosc: False',
    'report' : 'Opcja wyswietlania podsumowania testu. Domyslnie wartosc: F alse',
}

# wyodrebnia nazwe klasy z nazwy folderu (dla Stanford Dogs dataset)
def rename_class(name):
     return" ".join(name.split("-")[1].split("_"))

# funkcja zwraca transform dla danych treningowych
def get_train_transform():
    return transforms.Compose([transforms.Resize((224,224)),    # przetwarzamy zdjecia przeskalowane na (224,224)
                                      transforms.RandomHorizontalFlip(),  # losowe lustrzane odbicie w poziomie
                                      transforms.RandomResizedCrop(
                                          (224, 224), scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # losowe przyciecie
                                      transforms.ToTensor(),        # dane jako tensor
                                      transforms.Normalize(
                                          DATA_MEANS, DATA_STD)
                                      ])

# funkcja zwraca transform dla danych testowych
def get_test_transform():
    return transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         DATA_MEANS, DATA_STD)
                                     ])

# funkcja mapuje klasy na podstawie glownego zbioru danych (mamy taki sam seed wiec funkcja zawsze zwroci pasujacy slownik)
def map_classes():
    base_path = os.getcwd() +'/dataset/Images'
    classes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    classes.sort()
    folder_mapping = {class_name : index for index, class_name in enumerate(classes)}   # wartosc zwracana to slownik {nazwa_klasy : indeks_klasy)
    return folder_mapping

# funkcja zwraca model do kodowania danych do embedded space (tutaj zmniejszony Densenet)
def get_backbone(output_size):
    encoder = torchvision.models.DenseNet(growth_rate=32,
                                          block_config=(6, 6, 6, 6),
                                          bn_size=2,
                                          num_init_features=64,
                                          num_classes=output_size  
                                         )
    return encoder

# rozdzielamy batch na subsety: support i query
def split_batch(imgs, targets):
    support_imgs, query_imgs = imgs.chunk(2, dim=0)
    support_targets, query_targets = targets.chunk(2, dim=0)
    return support_imgs, query_imgs, support_targets, query_targets

class BatchSampler(object):
    '''
    Klasa do tworzenia zadan treningowych.
    Wejscia:
        - group_targets - tensor z etykietami danych w subsecie treningowym lub walidacyjnym
        - Group_count - liczba klas w zadaniu
        - Sample_count - liczba danych dla kazdej klasy
        - double_support - dodaje subset query o takiej samej wielkosci jak support set (wygodniejsze przesylanie danych)
        - randomize - losowo przeklada dane
    '''
    def __init__(self, group_targets, Group_count, Sample_count, double_support=False, randomize=True):
        self.group_targets = group_targets
        self.Group_count = Group_count
        self.Sample_count = Sample_count
        self.randomize = randomize
        self.include_extra = double_support
        if self.include_extra:
            self.Sample_count *= 2
        self.group_size = self.Group_count * self.Sample_count  

        self.group_ids = torch.unique(self.group_targets).tolist()
        self.total_groups = len(self.group_ids)
        self.samples_per_group = {}
        self.groups_per_sample = {}  
        for g in self.group_ids:
            self.samples_per_group[g] = torch.where(self.group_targets == g)[0]
            self.groups_per_sample[g] = self.samples_per_group[g].shape[0] // self.Sample_count

        self.total_iterations = sum(self.groups_per_sample.values()) // self.Group_count
        self.group_queue = [g for g in self.group_ids for _ in range(self.groups_per_sample[g])]
        if self.randomize:
            self.mix_data()
        else:
            ordered_indices = [i+q*self.total_groups for i, 
                               g in enumerate(self.group_ids) for q in range(self.groups_per_sample[g])]
            self.group_queue = np.array(self.group_queue)[np.argsort(ordered_indices)].tolist()

    def mix_data(self):
        for g in self.group_ids:
            permutation = torch.randperm(self.samples_per_group[g].shape[0])
            self.samples_per_group[g] = self.samples_per_group[g][permutation]
        random.shuffle(self.group_queue)

    def __iter__(self):
        if self.randomize:
            self.mix_data()

        starting_point = defaultdict(int)
        for iteration in range(self.total_iterations):
            current_group = self.group_queue[iteration*self.Group_count:(iteration+1)*self.Group_count]  
            sample_batch = []
            for g in current_group: 
                sample_batch.extend(self.samples_per_group[g][starting_point[g]:starting_point[g]+self.Sample_count])
                starting_point[g] += self.Sample_count
            if self.include_extra: 
                sample_batch = sample_batch[::2] + sample_batch[1::2]
            yield sample_batch

    def __len__(self):
        return self.total_iterations
    
class ImageDataset(Dataset):
    '''
    Klasa zbioru danych.
    Wejscia:
        - root_dir - sciezka do subsetu danych (train/test/val)
        - label_map - tensor etykiet klas
        - transformacja danych (do augmentacji)
    '''
    def __init__(self, root_dir, label_map, img_transform=None):
        super().__init__()
        self.img_transform = img_transform
        self.label_map = label_map
        self.img_labels = []
        self.img_paths = []

        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    self.img_paths.append(img_path)
                    self.img_labels.append(self.label_map[class_dir]) 

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        label = self.img_labels[idx]

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.img_paths)

    def get_labels_as_tensor(self):
        return torch.tensor(self.img_labels, dtype=torch.long)
    
class PrototypicalNetwork(pl.LightningModule):
    '''
    Klasa sieci prototypowej.
    Hiperparametry:
        - feature_dim - wymiar embedding space
        - learning_rate - wspolczynnik uczenia
    '''
    def __init__(self, feature_dim, learning_rate):

        super().__init__()
        self.save_hyperparameters()
        self.model = get_backbone(output_size=self.hparams.feature_dim)                 # pobieramy model do kodowania danych

    def configure_optimizers(self):                                                     # ustawiamy optymalizator (scheduler  zmienia learning rate w przypadku braku poprawy podczas treningu)
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]

    @staticmethod
    def calculate_prototypes(features, targets):                                        # funkcja do obliczania prototypow klas
        classes, _ = torch.unique(targets).sort()                                       # sprawdzamy jakie mamy klasy
        prototypes = []
        for c in classes:
            p = features[torch.where(targets == c)[0]].mean(dim=0)                      # srednia z wektorow klas
            prototypes.append(p)
        prototypes = torch.stack(prototypes, dim=0)
                                                                                        # zwracamy klasy odpowiadajace centroidom
        return prototypes, classes

    def classify_feats(self, prototypes, classes, feats, targets):                      # funkcja do klasyfikacji
        
        dist = torch.pow(prototypes[None, :] - feats[:, None], 2).sum(dim=2)            # odleglosc euklidesowa podniesiona do kwadratu
        preds = F.log_softmax(-dist, dim=1)                                             # predykcja na podstawie loglikelihood z zanegowanych dystansow
        labels = (classes[None, :] == targets[:, None]).long().argmax(dim=-1)
        acc = (preds.argmax(dim=1) == labels).float().mean()
        return preds, labels, acc

    def calculate_loss(self, batch, mode):                                              # obliczamy strate 

        imgs, targets = batch
        features = self.model(imgs)                                                                     # wyciagamy cechy ze zdjec z uzyciem modelu do kodowania danych
        support_feats, query_feats, support_targets, query_targets = split_batch(features, targets)     # rozdzielamy na support set i query set
        prototypes, classes = PrototypicalNetwork.calculate_prototypes(support_feats, support_targets)  # obliczamy prototypy
        preds, labels, acc = self.classify_feats(prototypes, classes, query_feats, query_targets)       # dokonujemy klasyfikacji
        loss = F.cross_entropy(preds, labels)                                                           # obliczamy wartosc straty

        self.log(f"{mode}_loss", loss)                                                                  # zapisujemy loss i accuracy do logow
        self.log(f"{mode}_acc", acc)
        return loss

    def training_step(self, batch, batch_idx):                                                          # krok treningowy (obliczamy strate w fazie treningu)
        return self.calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):                                                        # krok walidacyjny (obliczamy strate w fazie walidacji)
        _ = self.calculate_loss(batch, mode="val")

class ImageViewer(tk.Tk):
    '''
    Klasa prostego GUI do wyboru zdjecia do pojedynczej predykcji. 
    '''
    def __init__(self):
        super().__init__()
        self.cwd = os.getcwd() + '/imgs'
        self.selected_image_path = None     
        self.title("Image Previewer")
        self.geometry("600x400")
        
        self.listbox = tk.Listbox(self, width=20)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.preview_image)
        
        self.files = os.listdir(self.cwd)
        for file in self.files:
            self.listbox.insert(tk.END, file)

        self.image_label = tk.Label(self, width=50)
        self.image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.ok_button = tk.Button(self, text="OK", command=self.save_path)
        self.ok_button.pack(side=tk.BOTTOM, fill=tk.X)

        self.cancel_button = tk.Button(self, text="Cancel", command=self.destroy)
        self.cancel_button.pack(side=tk.BOTTOM, fill=tk.X)
    
    def preview_image(self, event):
        
        # funkcja do wyswietlania podgladu wybranego zdjecia

        index = self.listbox.curselection()
        if index:
            file_name = self.listbox.get(index)
            
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):   # wyswietlamy rozne formaty zdjec
                
                image_path = os.path.join(self.cwd, file_name)
                img = Image.open(image_path)                                            # wczytujemy zdjecie w pillow
                img.thumbnail((200, 200))  
                photo = ImageTk.PhotoImage(img)                                         # renderujemy zdjecie w GUI
                self.image_label.config(image=photo, anchor='center')
                self.image_label.image = photo  
            else:

                self.image_label.config(image='')                                       # do poki nie wybierzemy zdjecia nie wyswietlaj nic
                self.image_label.image = None
    
    def save_path(self):                                                                # zapisujemy sciezke do wybranego zdjecia (do pozniejszego wczytania zdjecia do moedlu)
        index = self.listbox.curselection()
        if index:
            file_name = self.listbox.get(index)
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                self.selected_image_path = os.path.join(self.cwd, file_name)
                self.quit()



