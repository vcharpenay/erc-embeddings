from pykeen.pipeline import pipeline
from pykeen.losses import MarginRankingLoss, NSSALoss, CrossEntropyLoss
from datasets import FB15k237_ET
from base.joint_training_loop import JointSLCWATrainingLoop

import octagons
import octagons_ensemble
import octagons_dist

# m = octagons.OctagonsModel
# m = octagons_ensemble.OctagonsModel
m = octagons_dist.OctagonsModel

ds = FB15k237_ET()

dim = 64

# l = MarginRankingLoss
l = NSSALoss
# l = CrossEntropyLoss

result = pipeline(
    dataset=ds,
    model=m,
    model_kwargs=dict(embedding_dim=dim),
    training_kwargs=dict(num_epochs=50),
    loss=l,
    training_loop=JointSLCWATrainingLoop
)
    
result.save_to_directory(f"results/{dim}")