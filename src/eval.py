from pykeen.pipeline import pipeline
from pykeen.losses import MarginRankingLoss, NSSALoss
from datasets import FB15k237_ET
from octagons import OctagonsModel
from base.joint_training_loop import JointSLCWATrainingLoop

ds = FB15k237_ET()

dim = 10

l = MarginRankingLoss
# l = NSSALoss

result = pipeline(
    dataset=ds,
    model=OctagonsModel,
    model_kwargs=dict(embedding_dim=dim),
    training_kwargs=dict(num_epochs=5),
    loss=l,
    training_loop=JointSLCWATrainingLoop
)
    
result.save_to_directory(f"results/{dim}")