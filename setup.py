import os

from llm.modules.llm_interface import LLMInterface
from dotenv import load_dotenv

# important to access the access key of huggingface
load_dotenv()


def _is_model_available_locally(model_name):
    ''' Checks, if the given model is already part of the system. '''
    local_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_path = os.path.join(local_cache_dir, f"models--{model_name.replace('/', '--')}")
    
    return os.path.exists(model_path)


if __name__ == '__main__':
    """
    Allows us to setup the llm model.
    """
    model_name = 'meta-llama/Llama-3.2-3B-Instruct'

    if not _is_model_available_locally(model_name):
        print(f'Need to download the model {model_name}\nIt takes a while..')
        LLMInterface(model_name, print)
    else:
        print(f'The model {model_name} is already installed.')