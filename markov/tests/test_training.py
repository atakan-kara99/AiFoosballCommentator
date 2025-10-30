# make import from sibling directory work
import sys, os
sys.path.append(os.path.abspath('..'))

from src.pipelines import TrainingPipeline

path_to_autoencoder = "tests/test_contents/autoencoder.pickle"
path_to_model = "tests/test_contents/model.pickle"
path_to_statistics = "tests/test_contents/stats.txt"
path_to_touchlog = "tests/test_contents/touches.json"
path_to_eventlog = "tests/test_contents/eventlog.json"


p = TrainingPipeline(path_to_autoencoder, path_to_model, path_to_touchlog)

p.execute()