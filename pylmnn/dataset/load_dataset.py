import os.path
import json
from PIL import Image
import numpy as np


class TablePrinter(object):
    "Print a list of dicts as a table"

    def __init__(self, fmt, sep=' ', ul=None):
        """        
        @param fmt: list of tuple(heading, key, width)
                        heading: str, column label
                        key: dictionary key to value to print
                        width: int, column width in chars
        @param sep: string, separation between columns
        @param ul: string, character to underline column label, or None for no underlining
        """
        super(TablePrinter, self).__init__()
        self.fmt = str(sep).join('{lb}{0}:{1}{rb}'.format(key, width, lb='{', rb='}') for heading, key, width in fmt)
        self.head = {key: heading for heading, key, width in fmt}
        self.ul = {key: str(ul) * width for heading, key, width in fmt} if ul else None
        self.width = {key: width for heading, key, width in fmt}

    def row(self, data):
        return self.fmt.format(**{k: str(data.get(k, ''))[:w] for k, w in self.width.items()})

    def __call__(self, dataList):
        _r = self.row
        res = [_r(data) for data in dataList]
        res.insert(0, _r(self.head))
        if self.ul:
            res.insert(1, _r(self.ul))
        return '\n'.join(res)


def _dataset_verification(data_path):
    """
    check if data_path folder contain the following files
    ===============================================================================
    ├── classification_all.json
    └── images
        ├── image_0.jpg
        ├── image_1.jpg
        ├── image_2.jpg
    ===============================================================================
    """
    labels_path = os.path.join(data_path, 'classification_all.json')
    images_folder_path = os.path.join(data_path, 'images')
    if not os.path.exists(labels_path):
        print("label file not found ")
    if not os.path.exists(images_folder_path):
        print("image folder not found ")

    image_dict = {}
    label2id = {}

    max_id = 0
    with open(labels_path, 'r') as file_handle:
        label_file = json.load(file_handle, encoding='utf-8-sig')
        for i in label_file:
            # verify the existence of the image file
            im_name = i['image'].strip()
            images_path = os.path.join(images_folder_path, im_name)
            if not os.path.exists(images_path):
                print("image %s not found, its label discarded " % im_name)
                continue

            label = i['label'].strip()
            if label not in label2id:
                label2id[label] = max_id
                image_dict[label] = [images_path]
                max_id += 1
            else:
                image_dict[label].append(images_path)

    image_dict = {k: v for k, v in sorted(image_dict.items(), key=lambda item: item[0])}
    label2id = {k: v for k, v in sorted(label2id.items(), key=lambda item: len(image_dict[item[0]]))}

    print("sample statistics: ")
    data = []
    for label, idx in label2id.items():
        entry = {'labelid': idx, 'labelname': label, 'numofsamples': len(image_dict[label])}
        data.append(entry)

    fmt = [
        ('LabelID', 'labelid', 6),
        ('LabelName', 'labelname', 30),
        ('# samples', 'numofsamples', 20)
    ]

    print(TablePrinter(fmt, ul='=')(data))

    return image_dict, label2id


def _keep_dataset_with_enough_samples(image_dict, num_samples):
    np.random.seed(0)
    out_dataset = {}
    for k, v in image_dict.items():
        l_v = len(v)
        if l_v < num_samples:
            continue

        out_dataset[k] = v

    out_dataset = {k: v for k, v in sorted(out_dataset.items(),
                                           key=lambda item: item[0])}

    print("sample statistics: ")
    stats = []
    for label in out_dataset.keys():
        entry = {'label': label, 'num of samples': len(out_dataset[label])}
        stats.append(entry)

    fmt = [
        ('Label', 'label', 30),
        ('# of samples', 'num of samples', 15)
    ]

    print(TablePrinter(fmt, ul='=')(stats))

    return out_dataset


def _normalize_dataset(image_dict, num_samples):
    np.random.seed(0)
    balanced_dataset = {}
    for k, v in image_dict.items():
        l_v = len(v)
        if l_v < num_samples:
            continue
        # random sample
        select_idx = np.random.choice([i for i in range(l_v)],
                                      size=num_samples, replace=False)

        balanced_dataset[k] = [v[idx] for idx in range(l_v)
                               if idx in select_idx]

    balanced_dataset = {k: v for k, v in sorted(balanced_dataset.items(),
                                                key=lambda item: item[0])}

    print("sample statistics: ")
    stats = []
    for label in balanced_dataset.keys():
        entry = {'label': label, 'num of samples': len(balanced_dataset[label])}
        stats.append(entry)

    fmt = [
        ('Label', 'label', 30),
        ('# of samples', 'num of samples', 15)
    ]

    print(TablePrinter(fmt, ul='=')(stats))

    return balanced_dataset


def _split_dataset_by_ratio(image_dict, ratio):
    np.random.seed(3)

    data = {
        'train': {},
        'test': {}
    }

    for k, v in image_dict.items():
        l_v = len(v)
        train_len = int(l_v * ratio)

        # random sample
        select_idx = np.random.choice([i for i in range(l_v)],
                                      size=train_len, replace=False)

        data['train'][k] = [image_dict[k][idx] for idx in range(l_v)
                            if idx in select_idx]

        data['test'][k] = [image_dict[k][idx] for idx in range(l_v)
                           if idx not in select_idx]

    data['train'] = {k: v for k, v in sorted(data['train'].items(), key=lambda item: item[0])}
    data['test'] = {k: v for k, v in sorted(data['test'].items(), key=lambda item: item[0])}

    print("sample statistics: ")
    stats = []
    for label in data['train'].keys():
        entry = {'label': label, 'train samples': len(data['train'][label]),
                 'test samples': len(data['test'][label])}
        stats.append(entry)

    fmt = [
        ('Label', 'label', 30),
        ('Train', 'train samples', 15),
        ('Test', 'test samples', 15)
    ]

    print(TablePrinter(fmt, ul='=')(stats))

    return data


def _dataset_split_fewshot(image_dict, sample_thresh):
    data = {
        'train': {},
        'test': {}
    }

    for label in image_dict.keys():
        if len(image_dict[label]) < sample_thresh * 2:
            print("label %s has too few samples, thus discarded " % label)
            continue
        else:
            num_samples = len(image_dict[label])
            np.random.seed(0)
            train_idx = np.random.choice([i for i in range(num_samples)],
                                         size=sample_thresh, replace=False)
            assert len(train_idx) == len(set(train_idx))

            data['train'][label] = [image_dict[label][idx] for idx in range(num_samples)
                                    if idx in train_idx]

            data['test'][label] = [image_dict[label][idx] for idx in range(num_samples)
                                   if idx not in train_idx]

    data['train'] = {k: v for k, v in sorted(data['train'].items(), key=lambda item: item[0])}
    data['test'] = {k: v for k, v in sorted(data['test'].items(), key=lambda item: item[0])}

    print("sample statistics: ")
    stats = []
    for label in data['train'].keys():
        entry = {'label': label, 'train samples': len(data['train'][label]),
                 'test samples': len(data['test'][label])}
        stats.append(entry)

    fmt = [
        ('Label', 'label', 30),
        ('Train', 'train samples', 15),
        ('Test', 'test samples', 15)
    ]

    print(TablePrinter(fmt, ul='=')(stats))

    return data


def _trim_dataset(data, train_at_most, test_at_most):
    np.random.seed(0)

    out_data = {
        'train': {},
        'test': {}
    }

    for k, v in data['train'].items():
        l_v = len(v)
        if l_v < train_at_most:
            out_data['train'][k] = v
        else:
            # random sample
            select_idx = np.random.choice([i for i in range(l_v)],
                                          size=train_at_most, replace=False)

            out_data['train'][k] = [data['train'][k][idx] for idx in range(l_v)
                                    if idx in select_idx]

    out_data['train'] = {k: v for k, v in sorted(out_data['train'].items(),
                                                 key=lambda item: item[0])}

    for k, v in data['test'].items():
        l_v = len(v)
        if l_v <= test_at_most:
            out_data['test'][k] = v
        else:
            # random sample
            select_idx = np.random.choice([i for i in range(l_v)],
                                          size=test_at_most, replace=False)

            out_data['test'][k] = [data['test'][k][idx] for idx in range(l_v)
                                   if idx in select_idx]

    out_data['test'] = {k: v for k, v in sorted(out_data['test'].items(),
                                                key=lambda item: item[0])}
    print("sample statistics: ")
    stats = []
    for label in out_data['train'].keys():
        entry = {'label': label, 'train samples': len(out_data['train'][label]),
                 'test samples': len(out_data['test'][label])}
        stats.append(entry)

    fmt = [
        ('Label', 'label', 30),
        ('Train', 'train samples', 15),
        ('Test', 'test samples', 15)
    ]

    print(TablePrinter(fmt, ul='=')(stats))

    return out_data


def _load_loreal_data(data_path, normalize_h=64, normalized_w=64):
    # check if data_path folder contain the following files
    # ===============================================================================
    # ├── classification_all.json
    # └── images
    #     ├── image_0.jpg
    #     ├── image_1.jpg
    #     ├── image_2.jpg
    # ===============================================================================
    labels_path = os.path.join(data_path, 'classification_all.json')
    images_path = os.path.join(data_path, 'images')
    if not os.path.exists(labels_path):
        print("label file not found ")
    if not os.path.exists(images_path):
        print("image folder not found ")

    X = []
    y = []

    label2id = {}
    max_id = 0
    with open(labels_path, 'r') as file_handle:
        label_file = json.load(file_handle, encoding='utf-8-sig')
        for i in label_file:
            im_name = i['image'].strip()
            label = i['label'].strip()
            if label not in label2id:
                label2id[label] = max_id
                max_id += 1

            y.append(label2id[label])
            # read file
            im = Image.open(os.path.join(images_path, im_name))
            im = np.array(im.resize((normalize_h, normalized_w)))
            X.append(im)

    X = np.stack(X, axis=0)
    y = np.asarray(y)

    return X, y, label2id


if __name__ == '__main__':
    data_path = '/Users/zyuan/Downloads/loreal_135_classification'
    image_dict, label2id = _dataset_verification(data_path)

    data = _dataset_split_fewshot(image_dict, 50)
    exit()
    X, y, label2id = _load_loreal_data(data_path)
    np.savez_compressed(os.path.join(data_path, 'loreal_135.npz'), X=X, y=y)
