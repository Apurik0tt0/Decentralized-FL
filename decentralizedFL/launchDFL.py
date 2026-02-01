import os
import sys


module_path = os.path.abspath(os.path.join('..'))
sys.path.insert(0, module_path)

from fluke.data.datasets import Datasets
dataset = Datasets.get("mnist", path="./data")

from fluke.data import DataSplitter
splitter = DataSplitter(dataset=dataset,
                        distribution="dir")

from fluke.evaluation import ClassificationEval
from fluke import FlukeENV

evaluator = ClassificationEval(eval_every=1, n_classes=dataset.num_classes)
FlukeENV().set_evaluator(evaluator)
FlukeENV().set_seed(42)

from fluke import DDict
client_hp = DDict(
    batch_size=10,
    local_epochs=5,
    loss="CrossEntropyLoss",
    optimizer=DDict(
      name="SGD",
      lr=0.01,
      momentum=0.9,
      weight_decay=0.0001),
    scheduler=DDict(
      gamma=1,
      step_size=1)
)

# we put together the hyperparameters for the algorithm
hyperparams = DDict(client=client_hp,
                    server=DDict(weighted=True),
                    model="MNIST_2NN")

# import your own algorithm here
from p2p import P2P
algorithm = P2P(7, splitter, hyperparams)

from fluke.utils.log import Log
logger = Log()
algorithm.set_callbacks(logger)

algorithm.run(10, 0.5)