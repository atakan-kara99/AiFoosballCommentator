"""
LLM Interface Test Environment
===========================

This module provides an interactive test environment for the LLM Interface,
allowing manual testing of model responses and performance metrics.

Features:
--------
1. Model Selection:
   - Default model: meta-llama/Llama-3.2-3B-Instruct

2. Interactive Testing:
   - Real-time prompt testing
   - Response generation
   - Performance timing

3. Error Handling:
   - Input validation
   - Model exceptions

Usage:
-----
1. Run the script
2. Select model (or use default)
3. Enter prompts to test
4. View responses and timing
5. Type 'q' to exit
"""

# Standard library imports
import sys
import os
import time

# Add parent directory to Python path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules import llm_interface as llm


# Visual separator for console output
def print_header():
    """
    Display formatted header for test environment.
    Provides visual separation and context in console output.
    """
    print("--------------------------------------------")
    print("|                  TEST                    |")
    print("|              LLM Interface               |")
    print("--------------------------------------------")
    print("")


# Initialize test environment
print_header()

# Model selection with default fallback
model = input("Enter the model you want to test (or press ENTER to use \"meta-llama/Llama-3.2-3B-Instruct\"): \n")
if model == "":
    model = "meta-llama/Llama-3.2-3B-Instruct"

# Initialize LLM interface with selected model
llm = llm.LLMInterface(model=model)

# Interactive test loop
while True:
    # Get user prompt
    user_prompt = input("Enter your prompt (type 'q' to quit): \n")
        
    # Check for exit condition
    if user_prompt.lower() == "q":
        print("\nExiting LLM Interface Test Environment. \n")
        break

    # Generate and time response
    try:
        # Start performance timer
        start_time = time.time()
        
        # Generate response
        # Note: Uncomment the following line for extended testing
        # response = llm.test_generate_comment(user_prompt)

        # Standard response generation
        response = llm.generate_comment(user_prompt)

        # Calculate and format generation time
        end_time = time.time()
        generation_time = end_time - start_time

        # Display results
        print(f"\nResponse ({generation_time:.2f}s): ")
        print(response[0]['generated_text'] + "\n")

    except Exception as e:
        # Handle and display any errors during generation
        print(f"An error occurred: {e}")