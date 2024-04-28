import os 
from shutil import copy2 
import shutil
from scipy.io import loadmat
from torch import randperm, manual_seed

def merge_folders(src1, src2, dst):
    if not os.path.exists(dst):                             # jesli nie ma folderu docelowego to tworzymy pusty folder
        os.makedirs(dst)
    
    subfolders = set(os.listdir(src1) + os.listdir(src2))   # foldery z klasami

    for folder in subfolders:
        dst_path = os.path.join(dst, folder)                # tworzymy folder dla kazdej klasy w docelowym folderze
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        for src in [src1, src2]:                            # kopiujemy zawartosc dla stworzonego folderu klasy
            src_path = os.path.join(src, folder)
            if os.path.exists(src_path):
                for file in os.listdir(src_path):
                    shutil.copy(os.path.join(src_path, file), dst_path)

    path_to_rm = '/'.join(src1.split("/")[:-1])             # pozbywamy sie buforowych folderow (linia 121)
    shutil.rmtree(path_to_rm)

def read_mat_file(mat_file: str, num_classes, num_files):

    # funckja odczytuje plik .mat, zwraca slownik uzywanych plikow

    mat_info = loadmat(mat_file) 

    file_list = mat_info["file_list"] 
    print('[1] file_list: ', file_list, type(file_list))

    print('[2] file_list[0, 0][0]: ', str(file_list[0, 0][0]), type(str(file_list[0, 0][0])))

    dic_of_used_files = {} 
    cnt = 0 
    for id, file in enumerate(file_list): 
        cur_class, file_path = file[0][0].split("/") 
        if cur_class not in dic_of_used_files: 
            cnt += 1
            if cnt > num_classes: 
                break
            dic_of_used_files[cur_class] = [] 
        if len(dic_of_used_files[cur_class]) < num_files: 
            dic_of_used_files[cur_class].append(file_path)

    return dic_of_used_files

def split_data_into_dirs(input_data_path, output_data_path, num_classes, train_samples, test_samples):

    # funkcja na podstawie plikow .mat rozdziela zdjecia na train i test set, przydziela okreslona ilosc danych na kazda klase w danym subsecie

    input_images_path = os.path.join(input_data_path, "Images")
    input_lists_path = os.path.join(input_data_path, "lists")
    train_mat_file_path = os.path.join(input_lists_path, "train_list.mat")
    test_mat_file_path = os.path.join(input_lists_path, "test_list.mat")

    train_dic_of_used_files = read_mat_file(train_mat_file_path, num_classes, train_samples) 
    test_dic_of_used_files = read_mat_file(test_mat_file_path, num_classes, test_samples) 
    class_names = train_dic_of_used_files.keys() 


    os.makedirs(output_data_path, exist_ok=True)
    os.makedirs(os.path.join(output_data_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_data_path, 'test'), exist_ok=True)


    for class_name in class_names:
        os.makedirs(os.path.join(output_data_path, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(output_data_path, 'test', class_name), exist_ok=True)


    for class_name, list_of_files in train_dic_of_used_files.items():
        for file_name in list_of_files:
            in_path = os.path.join(input_images_path, class_name, file_name)
            out_path = os.path.join(output_data_path, "train", class_name, file_name)
            copy2(in_path, out_path)

    for class_name, list_of_files in test_dic_of_used_files.items():
        for file_name in list_of_files:
            in_path = os.path.join(input_images_path, class_name, file_name)
            out_path = os.path.join(output_data_path, "test", class_name, file_name)
            copy2(in_path, out_path)

def move_folders_to_datasets(base_path, train_classes, val_classes, test_classes):

    # funkcja 

    classes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]   # odczytujemy klasy z datasetu
    classes.sort()
    folder_mapping = {index : class_name for index, class_name in enumerate(classes)}           # mapujemy klasy na indeksy


    def move_folders(class_indices, target_dir):    # funkcja pomocnicza do przenoszenia folderow
        for index in class_indices:
            folder_name = folder_mapping.get(int(index))
            if folder_name:
                src_path = os.path.join(base_path, folder_name)
                dst_path = os.path.join(base_path, target_dir, folder_name)
                if os.path.exists(src_path):
                    shutil.move(src_path, dst_path)
                else:
                    print(f"Folder {src_path} does not exist and cannot be moved.")
            else:
                print(f"No folder mapped to index {index}.")


    for dir_name in ['train', 'val', 'test']:           # tworzymy subsety 
        dir_path = os.path.join(base_path, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    move_folders(train_classes, 'train')                # rozdzielamy klasy na podstawie wylosowanych klas
    move_folders(val_classes, 'val')
    move_folders(test_classes, 'test')


pwd = os.getcwd()                           #sciezka do folderu w ktorym pracujemy
base_path = pwd +'/Dataset/'                #sciezka do pobranego zbioru danych

input_path = pwd + '/dataset'
output_path = pwd + '/train_val_test' 

split_data_into_dirs(input_path, output_path, 120, 100, 50)                 #funkcja bierze 150 zdjec dla kazdej klasy (100-train, 50-test)
merge_folders(f'{output_path}/train', f'{output_path}/test', base_path)     #funkcja laczy train i test set (teraz mamy tyle samo zdjec dla kadzej klasy w jednym folderze)

manual_seed(0)                                                              #ustawiamy seed zeby za kazdym razem losowanie bylo takie samo
classes = randperm(120)                                                     #losujemy klasy
train_classes, val_classes, test_classes = classes[:96], classes[96:108], classes[108:] #rozdzielamy dataset 8:1:1

move_folders_to_datasets(base_path, train_classes, val_classes, test_classes) #tworzymy nowy dataset w oparciu o wylosowane klasy


