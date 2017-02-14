__sets = {}


from datasets.custom_dataset import custom_dataset
import numpy as np

devkit_path = ""
for split in ['train', 'test']:
    name = '{}_{}'.format('inria', split)
    __sets[name] = (lambda split=split: custom_dataset(split, devkit_path))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()