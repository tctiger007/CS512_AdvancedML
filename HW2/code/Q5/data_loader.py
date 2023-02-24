import gzip
import csv
import numpy as np

class DataLoader:
    def __init__(self):
        data_path = '/content/gdrive/My Drive/hw2/data/letter.data.gz'
        lines = self._read(data_path)
        self.data, self.target,self.nextletter = self._parse(lines)

    @staticmethod
    def _read(filepath):
        with gzip.open(filepath, 'rt') as file_:
            reader = csv.reader(file_, delimiter='\t')
            lines = list(reader)
            return lines

    @staticmethod
    def _parse(lines):
        lines = sorted(lines, key=lambda x: int(x[0]))
        data, target, nextletter = [], [],[]
        next_ = None

        for line in lines:
            nextletter.append(int(line[2]))
            pixels = np.array([int(x) for x in line[6:134]])
            pixels = pixels.reshape((16, 8))
            data.append(pixels)
            target.append(line[1])
        return np.array(data), np.array(target),np.array(nextletter)


def get_dataset():
    dataset = DataLoader()
    # Shuffle order of examples.
    target = np.zeros(dataset.target.shape + (26,))
    for index, letter in np.ndenumerate(dataset.target):
        target[index][ord(letter) - ord('a')] = 1
    dataset.target = target

    order = np.random.permutation(len(dataset.data))
    dataset.data = dataset.data[order]
    dataset.target = dataset.target[order]
    dataset.nextletter = dataset.nextletter[order]
    return dataset
