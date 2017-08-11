import os
import csv


def file_reader():
    """File reader reads the complete folder and make a list of classes set the split ratio
    which devides data into training and testing data
    :return training: 1st half splitted from the ratio
    :return testing: 2nd half splitted from the ratio
    """
    training = []
    testing = []
    data_split_ratio = 0.98
    root_path = 'data/'
    # top_view file_names
    file_names = [file_name for file_name in os.listdir(root_path)]
    file_paths = [root_path + file_path for file_path in file_names]

    # Fault types
    class_titles = [file_name[:-4] for file_name in file_names]

    for file_path in file_paths:

        training, testing = load_data_set(file_path, data_split_ratio)

    return training, testing, class_titles


def load_data_set(filename, split_ratio, training_set=[], testing_set=[]):
    """load_data_set split the data into training and testing data using the ratio
    :return training_set
    :return testing_set
    """
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        iterator = 0
        for x in range(len(dataset) - 1):
            for y in range(len(dataset[x])-1):
                dataset[x][y] = float(dataset[x][y])
            if iterator < split_ratio*lines.line_num:
                training_set.append(dataset[x])
            else:
                testing_set.append(dataset[x])
            iterator += 1
    return training_set, testing_set

if __name__ == "__main__":
    file_reader()
