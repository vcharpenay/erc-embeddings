from numpy import genfromtxt
from pickle import load
from pykeen.datasets import Dataset
from pykeen.triples import CoreTriplesFactory
from torch import tensor, max

def num_entities(t):
    return max(max(t[:,0]),max(t[:,2])).item() + 1

def num_relations(t):
    return max(t[:,1]).item() + 1

def num_classes(seq):
    return len(set().union(*seq))

def shift(seq, offset):
    return { i - offset for i in seq }

class FB15k237_ET(Dataset):

    def __init__(self) -> None:
        super().__init__()

        f = open("datasets/FB15K237-ET/ent2classes.pkl", "r+b")
        # note: ID 14496 isn't used in dataset
        types = [ shift(cs, 14497) for cs in load(f).values() ]
        f.close()

        train = tensor(genfromtxt("datasets/FB15K237-ET/train2id.txt", delimiter="\t", dtype=int))
        valid = tensor(genfromtxt("datasets/FB15K237-ET/valid2id.txt", delimiter="\t", dtype=int))
        test = tensor(genfromtxt("datasets/FB15K237-ET/test2id.txt", delimiter="\t", dtype=int))

        self.training = CoreTriplesFactory(
            mapped_triples=train,
            num_entities=num_entities(train),
            num_relations=num_relations(train),
            metadata=dict(mapped_types=types,num_classes=num_classes(types))
        )

        self.validation = CoreTriplesFactory(
            mapped_triples=valid,
            num_entities=num_entities(train),
            num_relations=num_relations(train)
        )

        self.testing = CoreTriplesFactory(
            mapped_triples=test,
            num_entities=num_entities(train),
            num_relations=num_relations(train)
        )