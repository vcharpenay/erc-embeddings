from typing import Sequence
from class_resolver import HintOrType, OptionalKwargs
from class_resolver.utils import OneOrManyHintOrType, OneOrManyOptionalKwargs
from pykeen.typing import HeadRepresentation, RelationRepresentation, TailRepresentation
from pykeen.triples import KGInfo
from pykeen.models import ERModel
from pykeen.models.nbase import _prepare_representation_module_list
from pykeen.nn.modules import Interaction, interaction_resolver
from pykeen.nn.representation import Representation

class ERCModel(ERModel):

    class_representations: Sequence[Representation]

    def __init__(
        self,
        *,
        triples_factory: KGInfo,
        skip_checks: bool = False,
        triple_interaction: HintOrType[Interaction[HeadRepresentation, RelationRepresentation, TailRepresentation]],
        type_interaction: HintOrType[Interaction],
        # interaction_kwargs: OptionalKwargs = None,
        # entity_representations: OneOrManyHintOrType[Representation] = None,
        # entity_representations_kwargs: OneOrManyOptionalKwargs = None,
        # relation_representations: OneOrManyHintOrType[Representation] = None,
        # relation_representations_kwargs: OneOrManyOptionalKwargs = None,
        class_representations,
        class_representations_kwargs,
        **kwargs
    ):
        """
        :param class_representations:
            representations of classes
        :param class_representations_kwargs:
            keyword arguments for creating class representations
        """
        super().__init__(
            triples_factory=triples_factory,
            interaction=triple_interaction,
            # interaction_kwargs=interaction_kwargs,
            # entity_representations=entity_representations,
            # entity_representations_kwargs=entity_representations_kwargs,
            # relation_representations=relation_representations,
            # relation_representations_kwargs=relation_representations_kwargs,
            **kwargs
        )

        # self.triple_interaction = self.interaction
        self.type_interaction = interaction_resolver.make(type_interaction)

        self.class_representations = _prepare_representation_module_list(
            representations=class_representations,
            representations_kwargs=class_representations_kwargs,
            max_id=triples_factory.metadata["num_classes"],
            shapes=self.type_interaction.class_shape,
            label="class",
            skip_checks=skip_checks
        )

    def score_ec(self, ic_batch, *, mode = None):
        i = [ i(indices=ic_batch[:,0]) for i in self.entity_representations ]
        c = [ i(indices=ic_batch[:,1]) for i in self.class_representations ]

        return self.type_interaction.score_ec(
            i[0] if len(i) == 1 else i,
            c[0] if len(c) == 1 else c
        )