from PIL import Image
import torch.nn.functional as F
import torch
import torch.utils.data as data
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from utils import PrototypicalNetwork, map_classes, get_test_transform, ImageDataset, ImageViewer, rename_class, help_info
import argparse

# funkcja do przetworzenia wejsciowego zdjecia tak jak w przypadku reszty danych testowych
def preprocess_single_pred(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = get_test_transform()
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    
    return img

# funkcja do przeprowadzania pojdeynczej predykcji
@torch.no_grad()
def predict_single_image(neural_net, data_set, image_path, k_shot):
    neural_net = neural_net.to(device)
    neural_net.eval()
    class_count = data_set.get_labels_as_tensor().unique().shape[0]
    examples_per_class = data_set.get_labels_as_tensor().shape[0] // class_count 

    data_loader = data.DataLoader(data_set, batch_size=128, num_workers=0, shuffle=False, drop_last=False)

    feature_vectors = []
    target_labels = []
    for batch_images, labels in tqdm(data_loader, "Przetwarzanie danych", leave=False):     
        batch_images = batch_images.to(device)
        extracted_features = neural_net.model(batch_images)                                 # kodujemy dane w zbiorze testowym
        feature_vectors.append(extracted_features.detach().cpu())
        target_labels.append(labels)
    feature_vectors = torch.cat(feature_vectors, dim=0)
    target_labels = torch.cat(target_labels, dim=0)

    target_labels, index_order = target_labels.sort()
    target_labels = target_labels.reshape(class_count, examples_per_class).transpose(0, 1)  
    feature_vectors = feature_vectors[index_order].reshape(class_count, examples_per_class, -1).transpose(0, 1) # obliczamy prototypu klas w zbiorze testowym

    
    k_feature_vectors, k_labels = feature_vectors[:k_shot].flatten(0,1), target_labels[:k_shot].flatten(0,1)
    class_prototypes, prototype_labels = neural_net.calculate_prototypes(k_feature_vectors, k_labels)

    processed_img = preprocess_single_pred(image_path)                  # ladujemy zdjecie wejsciowe po sciezce
    img_feature_vector = neural_net.model(processed_img)                # kodujemy zdjecie
    img_feature_vector = img_feature_vector.detach().cpu()              # odlaczamy dane od GPU

    distance_matrix = torch.pow(class_prototypes[None, :] - img_feature_vector[:, None], 2).sum(dim=2)  # liczymy dystanse do prototypow

    prediction_scores = F.log_softmax(-distance_matrix, dim=1)          # liczymy prawdopodobienstwo przynaleznosci do wszytkich klas

    class_prediction_index = prediction_scores.argmax(dim=1)
    predicted_class = prototype_labels[class_prediction_index]          # sprawdzamy ktora klasa zostala przewidziana (najwieksze prawdopodobienstwo)

    return prediction_scores, prototype_labels, predicted_class

# funkcja do wyswietlania rozkladu prawdopodobienstwa oraz podgladu badanego zdjecia
def plot_single_prediction_distribution(preds, proto_classes, classes_map, image_path, predicted_class):

    probabilities = preds.squeeze().exp().cpu().numpy()

    class_name = image_path.split("/")[-1].split(".")[0]


    class_names = [rename_class(class_name) for class_name, value in classes_map.items() if value in proto_classes]

    predicted_class_name = str([rename_class(class_name) for class_name, value in classes_map.items() if value == int(predicted_class)][0])

    img = plt.imread(image_path)

    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)  
    plt.imshow(img)
    plt.axis('off')  
    plt.title(f'Klasa: {class_name}')

    plt.subplot(1, 2, 2) 
    plt.bar(class_names, probabilities, color='skyblue')
    plt.ylabel('Prawdopodobienstwo przynaleznosci')
    plt.xticks(rotation=45, ha="right")
    plt.title('Rozkład prawdopodobienstwa dla kazdej klasy')
    
    plt.subplots_adjust(bottom=0.2)  # Adjust the bottom spacing
    plt.tight_layout() 

    plt.show()


parser = argparse.ArgumentParser(description='Skrypt do przeprowadzania pojedynczych predykcji')

parser.add_argument('--path', type=str, help=help_info['path'])
parser.add_argument('-K', type=int, default=5, help=help_info['K'])

args = parser.parse_args()

K= args.K
model_path = args.path

app = ImageViewer()
app.mainloop()
selected_path = app.selected_image_path


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

protonet_model = PrototypicalNetwork.load_from_checkpoint(model_path)
protonet_model.eval()

classes_dict = map_classes()
test_set = ImageDataset("Dataset/test", label_map=map_classes(), img_transform=get_test_transform())

a,bb, c = predict_single_image(protonet_model, test_set, selected_path, k_shot=K)

plot_single_prediction_distribution(a, bb, classes_dict, selected_path, c)


