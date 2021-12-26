from pathlib import Path
import numpy as np
import random
import argparse
import os

# how to run:
# python create_list.py -df true -f2f true -fs true -nt true --name 4+0.txt

face_images_paths = {
    # '': '/home/user1/xuyi/FaceF/dataset/manipulated_sequences/DeepFakeDetection/c23/face_images/',
    'deepfakes': '/cluster/home/xuyi/xuyi/FF++/manipulated_sequences/Deepfakes/raw/face_images/',
    'face2face': '/cluster/home/xuyi/xuyi/FF++/manipulated_sequences/Face2Face/raw/face_images/',
    'faceswap': '/cluster/home/xuyi/xuyi/FF++/manipulated_sequences/FaceSwap/raw/face_images/',
    'neuraltexures': '/cluster/home/xuyi/xuyi/FF++/manipulated_sequences/NeuralTextures/raw/face_images/',
    'original': '/cluster/home/xuyi/xuyi/FF++/original_sequences/youtube/raw/face_images/'
}

train_name = 'full_datasets_list.txt'
test_name = 'four_dataset_train_list.txt'
whole_list_path = '/home/user1/xuyi/Deepfake-Detection/data_list/' + train_name
test_data_list = '/home/user1/xuyi/Deepfake-Detection/data_list/' + test_name


def dataset_decide(deepfakes_inc, face2face_inc, faceswap_inc, neuraltextures_inc, original_inc):
    dataset_include = []

    if deepfakes_inc == 'true':
        dataset_include.append(face_images_paths['deepfakes'])
    if face2face_inc == 'true':
        dataset_include.append(face_images_paths['face2face'])
    if faceswap_inc == 'true':
        dataset_include.append(face_images_paths['faceswap'])
    if neuraltextures_inc == 'true':
        dataset_include.append(face_images_paths['neuraltexures'])
    if original_inc == 'true':
        dataset_include.append(face_images_paths['original'])

    return dataset_include


def find_path(file_paths, train_list_name, test_list_name, val_list_name):
    file_name_lists = []
    for file_path in file_paths:
        file_path = Path(file_path)
        file_name_list = list(file_path.glob('**/*.png'))
        file_name_lists += file_name_list

    image_label = np.array(file_name_lists).reshape(-1, 1)
    label = np.ones((len(file_name_lists), 1), dtype=np.int8)
    images_labels = np.hstack((image_label, label))

    total_data = len(images_labels)
    test_len = int(total_data*0.75)

    for single in images_labels:
        single_path = str(single[0].absolute())
        if 'youtube' in single_path.split('/'):
            single_label = '0'
        elif 'Deepfakes' in single_path.split('/'):
            single_label = '1'
        elif 'Face2Face' in single_path.split('/'):
            single_label = '2'
        elif 'FaceSwap' in single_path.split('/'):
            single_label = '3'
        elif 'NeuralTextures' in single_path.split('/'):
            single_label = '4'
        else:
            single_label = '?'
            print("from which dataset?")

        content = single_path + ' ' + single_label + '\n'

        if 'youtube' in single_path.split('/'):
            if int(single_path.split('/')[-2]) < 600:
                with open(train_list_name, 'a+') as ff:
                    ff.write(content)
            elif int(single_path.split('/')[-2]) > 799:
                with open(test_list_name, 'a+') as ff:
                    ff.write(content)
            elif 599 < int(single_path.split('/')[-2]) < 800:
                with open(val_list_name, 'a+') as ff:
                    ff.write(content)
        else:
            if int(single_path.split('/')[-2][0:3]) < 600:
                with open(train_list_name, 'a+') as ff:
                    ff.write(content)
            elif int(single_path.split('/')[-2][0:3]) > 799:
                with open(test_list_name, 'a+') as ff:
                    ff.write(content)
            elif 599 < int(single_path.split('/')[-2][0:3]) < 800:
                with open(val_list_name, 'a+') as ff:
                    ff.write(content)

        # if int(single_path.split('/')[-2]) < 600 or int(single_path.split('/')[-2][0:3]) < 600:
        #     with open(train_list_name, 'a+') as ff:
        #         ff.write(content)
        # elif int(single_path.split('/')[-2]) > 799 or int(single_path.split('/')[-2][0:3]) > 799:
        #     with open(test_list_name, 'a+') as ff:
        #         ff.write(content)
        # elif 599 < int(single_path.split('/')[-2]) < 800 or 599 < int(single_path.split('/')[-2][0:3]) < 800:
        #     with open(val_list_name, 'a+') as ff:
        #         ff.write(content)



def select_random_files(whole_list, select_list, num):
    lines = open(os.getcwd() + '/data_list/middle.txt').read().splitlines()
    myline = random.choices(lines, k=num)

    with open(select_list, 'w') as output:
        for row in myline:
            output.write(str(row) + '\n')

    # f = open(select_list, "w+")


    # np.savetxt(select_list, myline, fmt='%s')


def main():
    args = parse.parse_args()
    deepfakes_inc = args.deepfakes
    face2face_inc = args.face2face
    faceswap_inc = args.faceswap
    neuraltextures_inc = args.neuraltextures
    original_inc = args.original
    list_number = args.number
    train_list_name = args.name_train
    test_list_name = args.name_test
    val_list_name = args.name_val

    dataset_include = dataset_decide(deepfakes_inc, face2face_inc, faceswap_inc, neuraltextures_inc, original_inc)
    find_path(dataset_include, train_list_name, test_list_name, val_list_name)
    # select_random_files(whole_list, select_list, list_number)


if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--deepfakes', '-df', type=str, default='false')
    parse.add_argument('--face2face', '-f2f', type=str, default='false')
    parse.add_argument('--faceswap', '-fs', type=str, default='false')
    parse.add_argument('--neuraltextures', '-nt', type=str, default='false')
    parse.add_argument('--original', '-o', type=str, default='true')
    parse.add_argument('--number', '-num', type=int, default=10)
    parse.add_argument('--name_train', type=str, default='train.txt')
    parse.add_argument('--name_val', type=str, default='val.txt')
    parse.add_argument('--name_test', type=str, default='test.txt')

    main()
