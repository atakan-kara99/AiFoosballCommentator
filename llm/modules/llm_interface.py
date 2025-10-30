"""
Language Model Interface Module
============================

This module provides a high-level interface to interact with Hugging Face's
transformer-based language models. It handles model initialization, token management,
and text generation with configurable parameters.

Key Features:
-----------
1. Model Management:
   - Dynamic model loading from Hugging Face
   - GPU acceleration when available

2. Text Generation:
   - Configurable generation parameters
   - Error handling and logging

3. Performance Optimization:
   - GPU/CPU detection and utilization
   - Automatic device mapping

Security:
--------
API access requires a valid Hugging Face token set in the HUGGINGFACE_TOKEN
environment variable. Never hardcode this token in the source code.
"""

# Standard library imports
import sys
import os
from dotenv import load_dotenv

# Load environment variables (including HUGGINGFACE_TOKEN)
load_dotenv()

# Add parent directory to Python path for config import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import GENERAL_SETTINGS, LLM_INTERFACE_CONFIG

# ML/DL framework imports
from transformers import AutoTokenizer
import transformers
import torch
import warnings


# System configuration
VERSION     = GENERAL_SETTINGS["VERSION"]    # System version for logging
DEBUG_MODE  = GENERAL_SETTINGS["DEBUG_MODE"] # Debug flag for verbose output

# Model access configuration
API_KEY     = LLM_INTERFACE_CONFIG["API_KEY"] # Hugging Face API key

# Text generation parameters
MAX_LENGTH          = LLM_INTERFACE_CONFIG["MAX_LENGTH"]          # Maximum total sequence length
MAX_NEW_TOKENS      = LLM_INTERFACE_CONFIG["MAX_NEW_TOKENS"]      # Maximum new tokens to generate
TEMPERATURE         = LLM_INTERFACE_CONFIG["TEMPERATURE"]         # Sampling temperature (creativity)
TOP_K               = LLM_INTERFACE_CONFIG["TOP_K"]               # Top-k sampling parameter
TOP_P               = LLM_INTERFACE_CONFIG["TOP_P"]               # Nucleus sampling threshold
EARLY_STOPPING      = LLM_INTERFACE_CONFIG["EARLY_STOPPING"]      # Stop on end token
RETURN_FULL_TEXT    = LLM_INTERFACE_CONFIG["RETURN_FULL_TEXT"]    # Include prompt in output
TRUNCATION          = LLM_INTERFACE_CONFIG["TRUNCATION"]          # Truncate long inputs


class LLMInterface:
    """
    Interface for interacting with transformer-based language models.

    This class provides a high-level API for text generation using Hugging Face
    transformer models. It handles model initialization, GPU acceleration,
    and provides both testing and production interfaces for text generation.

    Attributes:
    ----------
    pipeline : transformers.Pipeline
        Text generation pipeline configured with model parameters

    Key Features:
    -----------
    1. Automatic GPU Detection:
       - Utilizes available GPU resources
       - Falls back to CPU if no GPU available
       - Supports multi-GPU setups

    2. Secure Authentication:
       - Uses environment variables for API tokens
       - Validates token access before model loading
       - Provides clear error messages for auth issues
    """

    def __init__(self, model: str, log_func):
        """
        Initialize the language model interface.

        Sets up the text generation pipeline with the specified model
        and configures it for optimal performance on the available hardware.

        Args:
        ----
        model : str
            Hugging Face model identifier
        log_func : callable
            Function for logging initialization progress

        Raises:
        ------
        EnvironmentError
            If HUGGINGFACE_TOKEN is not set
        RuntimeError
            If model initialization fails
        """
        log_func("Initializing LLM...")
        log_func("Model:                       " + model)
        log_func("Cuda Version:                " + str(torch.version.cuda))
        log_func("GPU is available:            " + str(torch.cuda.is_available()))

        # Check GPU availability and capabilities
        if torch.cuda.is_available():
            log_func("Number of available GPUs:    " + str(torch.cuda.device_count()))
            gpu_info = "Available GPUs: " + ", ".join(f"{torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count()))
            log_func(gpu_info)
        else:
            log_func("No GPUs available.")

        # Validate Hugging Face token
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token:
            raise EnvironmentError("HUGGINGFACE_TOKEN environment variable is not set")

        # Initialize model pipeline
        model_id = model
        log_func(f"Loading model {model_id} from HuggingFace...")
        
        try:
            # Verify model access first
            log_func("Verifying model access...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                token=hf_token  # Using token instead of deprecated use_auth_token
            )
            log_func("Access token loaded successfully.")

            # Set up text generation pipeline
            log_func("Loading LLM (if you don't have the model already, this may take a few minutes)...")
            self.pipeline = transformers.pipeline(
                "text-generation",                      # Task type
                model=model_id,                         # Model identifier
                torch_dtype=torch.float16,              # Half-precision for efficiency
                device_map="auto",                      # Automatic device selection
                tokenizer=tokenizer,                    # Tokenizer instance
                pad_token_id=tokenizer.eos_token_id,    # End of sequence token
                token=hf_token,                         # Authentication token
                
                # Generation parameters
                max_new_tokens   = MAX_NEW_TOKENS,      # Output length limit
                temperature      = TEMPERATURE,         # Sampling temperature
                top_k            = TOP_K,               # Top-k filtering
                top_p            = TOP_P,               # Nucleus sampling
                early_stopping   = EARLY_STOPPING,      # Stop on end token
                return_full_text = RETURN_FULL_TEXT,    # Include input in output
                truncation       = TRUNCATION           # Input truncation
            )
        except Exception as e:
            error_msg = f"Failed to initialize LLM: {str(e)}"
            log_func(error_msg)
            if "401" in str(e):
                log_func("Authentication error - please check your HuggingFace token and model access")
            raise RuntimeError(error_msg) from e

    def test_generate_comment(self, prompt: str) -> str:
        """
        Test text generation with detailed logging.

        Generates text from a prompt while capturing and reporting
        warnings and intermediate states. Used for debugging and
        testing the model's behavior.

        Args:
        ----
        prompt : str
            Input text to generate from

        Returns:
        -------
        str
            Generated text response

        Notes:
        -----
        - Prints detailed progress information
        - Captures and reports all warnings
        - Uses pipeline's default parameters
        """
        print("Generating response for the prompt: \n" + prompt + "\n")

        # Warning capture context
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            # Generate text using pipeline
            result = self.pipeline(prompt)
            
            # Report any warnings
            if caught_warnings:
                print("Warnings occurred:")
                for warning in caught_warnings:
                    print(f" - {warning.message}")
            else:
                print("No warnings occurred.")
        print("")

        return result[0]["generated_text"]
    
    def generate_comment(self, prompt: str) -> str:
        """
        Generate text from a prompt.

        Production version of text generation without debug output.
        Uses the pipeline's default parameters configured during
        initialization.

        Args:
        ----
        prompt : str
            Input text to generate from

        Returns:
        -------
        str
            Generated text response
        """
        result = self.pipeline(prompt)
        return result[0]["generated_text"].replace("\n", "")
