# make import from sibling directory work
import sys, os
sys.path.append(os.path.abspath('..'))

from src.pipelines import PseudoLivePipeLine

path_to_autoencoder = "test_contents/autoencoder.pickle"
path_to_model = "test_contents/model.pickle"
path_to_statistics = "test_contents/stats.txt"
path_to_touchlog = "test_contents/touches_new.json"
path_to_eventlog = "test_contents/eventlog.json"


p = PseudoLivePipeLine(path_to_autoencoder, path_to_model, path_to_statistics, path_to_touchlog, path_to_eventlog)

p.execute()