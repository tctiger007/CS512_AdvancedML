import cv2
import numpy as np


def read_train_data(path):
    image_list = list()
    y_data_list = list()
    with open(path) as f:
        for line in f:
            line = line.strip()
            line = line.split(' ')
            # print(len(line))
            y_data = line[:5]
            for i in range(len(y_data)):
                if i != 1:
                    y_data[i] = int(y_data[i])
            y_data_list.append(y_data)
            
            image = line[5:]
            image = [float(i) for i in image]
            image_list.append(image)
            
    return image_list, y_data_list


def read_transform_file(path):
    transformations = list()
    with open(path) as f:
        for line in f:
            line = line.strip()
            line = line.split(' ')
            for i in range(len(line)):
                if i != 0:
                    line[i] = int(line[i])
            transformations.append(line)
    return transformations


def get_word_wise_indices(images_arr, y_data):
    index_of_indices = list()
    start_index = 0
    end_index = 0
    for i in range(len(y_data)):
        if y_data[i][2] == -1:
            end_index = i
            index_of_indices.append((start_index, end_index+1))
            start_index = end_index + 1
        end_index += 1
    return index_of_indices


def get_letter_indices_from_word_index(index_of_indices, word_index):
    print(word_index)
    return [i for i in range(index_of_indices[word_index[0]], index_of_indices[word_index[1]])]
        

def translate_image(image, coordinates):
    image = np.reshape(image, (16, 8))
    # print(image)
    rows, cols = image.shape
    T = np.float32([[1, 0, coordinates[0]], [0, 1, coordinates[1]]])
    translated_image = cv2.warpAffine(image, T, (cols, rows))
    assert(image.shape == translated_image.shape)
    # print(translated_image)
    # print(translated_image.shape)
    return translated_image.flatten()


def rotate_image(image, degree):
    image = np.reshape(image, (16, 8))
    # print(image)
    rows, cols = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),degree,1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    assert(image.shape == rotated_image.shape)
    # print(rotated_image)
    # print(rotated_image.shape)
    return rotated_image.flatten()


def make_transformations(images_arr, transform_list, index_of_indices):
    for transform in transform_list:
        # print(transform)
        # print(index_of_indices[transform[1]-1])
        start, end = index_of_indices[transform[1]-1]
        indices_to_take = [i for i in range(start, end)]
        # print(indices_to_take)
        # transform_images = np.take(images_arr, indices_to_take, axis=0)
        # print(transform_images.shape)
        for index in indices_to_take:
            if transform[0] == 't':
                images_arr[index] = translate_image(images_arr[index], (transform[2], transform[3]))
            elif transform[0] == 'r':
                images_arr[index] = rotate_image(images_arr[index], transform[2])
    return images_arr


def write_transformed_data(path, images_arr, y_data):
    with open(path, 'w') as f:
        for i in range(images_arr.shape[0]):
            str_y_data = ' '.join([str(y) for y in y_data[i]])
            str_image = ' '.join([str(pix) for pix in images_arr[i]])
            f.write(str_y_data + ' ' + str_image)
            f.write('\n')



if __name__ == "__main__":
    image_list, y_data_list = read_train_data('../data/train.txt')
    # print(image_list[:10], y_data_list[:10])
    images_arr = np.array(image_list)
    print(images_arr.shape)
    index_list = get_word_wise_indices(images_arr, y_data_list)
    transform_list = read_transform_file('../data/transform.txt')
    images_arr = make_transformations(images_arr, transform_list, index_list)
    # translate_image(images_arr[0], (3,3))
    # rotate_image(images_arr[0], 180)
    print(images_arr.shape)
    write_transformed_data('../data/train_transform.txt', images_arr, y_data_list)
    # np.savetxt('../data/train_transform.txt', images_arr.flatten())
