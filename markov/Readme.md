# Markov

## Overview

This module contains the code for Backend Knowledge Extraction as described in the report. Its main functions include error handling, statistical analysis, and event recognition. The module parses and analyzes input messages from the CV module, then forwards event objects to the LLM group.

## Directory Structure

### `resources/`
- **`configs/`**: Contains global constants, state configuration information, and labeled touch sequences for event recognition
- **`data/`**: Includes gameplay data (both synthetic and real) and generators for states and game sequences

### `src/`
- **`model/`**: Contains Markov model classes and entity classes used for modeling
- **`statistics/`**: Code for statistical analysis of games
- **`error_handling/`**: Handles errors in input and reconstructs the most probable sequence

### `tests/`
- Contains tests for different functionalities

## Usage

The general functionality of this module is defined in the `markov/src/pipelines.py` file, which contains multiple pipelines for different use cases.