from typing import Sequence, List, Optional, NamedTuple, Type
from random import sample
from class_resolver import HintOrType, OptionalKwargs
from pykeen.sampling import NegativeSampler
from pykeen.training import SLCWATrainingLoop
from pykeen.triples import CoreTriplesFactory
from pykeen.triples.instances import SLCWABatch, BatchedSLCWAInstances
from pykeen.typing import MappedTriples
from torch import FloatTensor, LongTensor, BoolTensor, tensor, cat, min, max
from torch.utils.data import DataLoader

MappedTypes = Type[Sequence[List]]

class JointSLCWABatch(NamedTuple):
    positives_hrt: LongTensor
    negatives_hrt: LongTensor
    masks_hrt: Optional[BoolTensor]
    positives_ec: LongTensor
    negatives_ec: LongTensor

class JointBatchedSLCWAInstances(BatchedSLCWAInstances):

    def __init__(
        self,
        mapped_triples: MappedTriples,
        mapped_types: MappedTypes,
        batch_size: int = 1,
        drop_last: bool = True,
        num_entities: int = None,
        num_relations: int = None,
        negative_sampler: HintOrType[NegativeSampler] = None, negative_sampler_kwargs: OptionalKwargs = None
    ):
        super().__init__(mapped_triples, batch_size, drop_last, num_entities, num_relations, negative_sampler, negative_sampler_kwargs)

        self.mapped_types = mapped_types

        self.classes = set().union(*mapped_types)

    def __getitem__(
        self,
        item: List[int]
    ) -> JointSLCWABatch:
        positive_hrt_batch = self.mapped_triples[item]
        negative_hrt_batch, masks_hrt = self.negative_sampler.sample(positive_batch=positive_hrt_batch)

        # TODO mix head/tail?
        heads = positive_hrt_batch[:,0]
        positives_ec_batch = tensor([
            [h, self._pick_type(h)]
            for h in heads
        ])
        negatives_ec_batch = tensor([
            [h, self._pick_corrupted_type(h)]
            for h in heads
        ])
        # TODO filter mask
        
        return JointSLCWABatch(
            positives_hrt=positive_hrt_batch,
            negatives_hrt=negative_hrt_batch,
            masks_hrt=masks_hrt,
            positives_ec=positives_ec_batch,
            negatives_ec=negatives_ec_batch
        )
    
    def _pick_type(self, e):
        classes = self.mapped_types[e]
        return sample(classes, 1)[0]
    
    def _pick_corrupted_type(self, e):
        classes = self.mapped_types[e]
        return sample(self.classes.difference(classes), 1)[0]

class JointSLCWATrainingLoop(SLCWATrainingLoop):

    def __init__(self, negative_sampler: HintOrType[NegativeSampler] = None, negative_sampler_kwargs: OptionalKwargs = None, **kwargs):
        super().__init__(negative_sampler, negative_sampler_kwargs, **kwargs)
    
    def _create_training_data_loader(self, triples_factory: CoreTriplesFactory, sampler: str, batch_size: int, drop_last: bool, **kwargs) -> DataLoader[JointSLCWABatch]:
        # TODO 
        kwargs.pop("shuffle")

        return DataLoader(
            dataset=JointBatchedSLCWAInstances(
                # TODO add inverse, see triples_factory.create_slcwa_instances()
                mapped_triples=triples_factory.mapped_triples,
                # TODO extend triples_factory instead
                mapped_types=triples_factory.metadata["mapped_types"],
                batch_size=batch_size,
                drop_last=drop_last,
                num_entities=triples_factory.num_entities,
                num_relations=triples_factory.num_relations,
                negative_sampler=self.negative_sampler,
                negative_sampler_kwargs=self.negative_sampler_kwargs,
                #sampler=sampler
            ),
            # disable automatic batching
            batch_size=None,
            batch_sampler=None,
            **kwargs,
        )

    def _process_batch(self, batch: JointSLCWABatch, start: int, stop: int, label_smoothing: float = 0, slice_size: int = None) -> FloatTensor:
        model = self.model
        mode = self.mode
        loss = self.loss

        # Slicing is not possible in sLCWA training loops
        if slice_size is not None:
            raise AttributeError("Slicing is not possible for sLCWA training loops.")

        # split batch
        positive_hrt_batch, negative_hrt_batch, positive_hrt_filter, positive_ic_batch, negative_ic_batch = batch
        # TODO class filtering

        # send to device
        positive_hrt_batch = positive_hrt_batch[start:stop].to(device=model.device)
        negative_hrt_batch = negative_hrt_batch[start:stop]
        if positive_hrt_filter is not None:
            positive_hrt_filter = positive_hrt_filter[start:stop]
            negative_hrt_batch = negative_hrt_batch[positive_hrt_filter]
            positive_hrt_filter = positive_hrt_filter.to(model.device)
        # Make it negative batch broadcastable (required for num_negs_per_pos > 1).
        negative_score_shape = negative_hrt_batch.shape[:-1]
        negative_hrt_batch = negative_hrt_batch.view(-1, 3)

        # Ensure they reside on the device (should hold already for most simple negative samplers, e.g.
        # BasicNegativeSampler, BernoulliNegativeSampler
        negative_hrt_batch = negative_hrt_batch.to(model.device)

        # a = positive_hrt_batch[:,0]
        # b = positive_hrt_batch[:,1]
        # c = positive_hrt_batch[:,2]
        # print(min(a))
        # print(max(a))
        # print(min(b))
        # print(max(b))
        # print(min(c))
        # print(max(c))

        # Compute negative and positive scores
        positive_hrt_scores = model.score_hrt(positive_hrt_batch, mode=mode)
        negative_hrt_scores = model.score_hrt(negative_hrt_batch, mode=mode).view(*negative_score_shape)

        # TODO check model is of type ERCModel
        positive_ic_scores = model.score_ec(positive_ic_batch, mode=mode)
        negative_ic_scores = model.score_ec(negative_ic_batch, mode=mode)

        positive_scores=cat((positive_hrt_scores, positive_ic_scores))
        negative_scores=cat((negative_hrt_scores, negative_ic_scores))

        return (
            loss.process_slcwa_scores(
                positive_scores=positive_scores,
                negative_scores=negative_scores,
                label_smoothing=label_smoothing,
                batch_filter=positive_hrt_filter,
                num_entities=model._get_entity_len(mode=mode),
            )
            + model.collect_regularization_term()
        )