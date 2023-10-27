"""Data provider"""

import torch
import torch.utils.data as data
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import os
import nltk
import numpy as np
import h5py


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        self.loc = data_path + '/'
        # load the raw captions
        self.captions = []

        for line in open(self.loc + '%s_caps.txt' % data_split, 'rb'):
            self.captions.append(line.strip())

        # comput the TFIDF of captions
        self.tfidfs = self.tfidf_compute(self.captions)

        # load the image features
        self.images = np.load(self.loc + '%s_ims.npy' % data_split)
        self.length = len(self.captions)

        #self.im_div = 5
        # rkiros data has redundancy in images, we divide by 5
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def tfidf_compute(self, corpus, eps=1e-6):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(X).toarray()
        return tfidf

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div
        image = torch.Tensor(self.images[img_id])  # torch.Size([36, 2048])
        caption = self.captions[index]
        vocab = self.vocab
        
        # get tfidf of caption and image
        sem_for_caption = self.tfidfs[index]
        sems_for_image = self.tfidfs[img_id * 5: (img_id + 1) * 5]

        # convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(caption.lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        return image, target, index, img_id, sem_for_caption, sems_for_image

    def __len__(self):
        return self.length

def collate_fn(data):
    """
    Build mini-batch tensors from a list of (image, caption, index, img_id) tuples.
    Args:
        data: list of (image, target, index, img_id) tuple.
            - image: torch tensor of shape (36, 2048).
            - target: torch tensor of shape (?) variable length.
    Returns:
        - images: torch tensor of shape (batch_size, 36, 2048).
        - targets: torch tensor of shape (batch_size, padded_length).
        - lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids, sem_for_caption, sems_for_image = zip(*data)

    # Merge images (convert tuple of 2D tensor to 3D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    # TFIDF
    #tfidf_for_caption = torch.Tensor(np.array(sem_for_caption))
    #tfidf_for_image = torch.Tensor(np.array(sems_for_image))
    tfidf_for_caption = torch.Tensor(np.vstack(sem_for_caption))  # Use np.vstack to stack arrays
    tfidf_for_image = torch.Tensor(np.concatenate(sems_for_image, axis=0))  # Use np.concatenate to concatenate arrays

    return images, targets, lengths, ids, tfidf_for_caption, tfidf_for_image


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=6):
    dset = PrecompDataset(data_path, data_split, vocab)  # load data

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt):
    # get the data path
    dpath = os.path.join(opt.data_path, data_name)

    # get the train_loader
    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers)
    # get the val_loader
    val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    100, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size, workers, opt):
    # get the data path
    dpath = os.path.join(opt.data_path, data_name)

    # get the test_loader
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                     100, False, workers)
    return test_loader