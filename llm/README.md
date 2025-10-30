# LLM Commentary System

This module provides real-time natural language commentary generation for foosball events using Large Language Models (LLM). The system processes game events, generates contextual prompts, and delivers natural-sounding commentary with support for interrupts and statistical insights.

## 📁 Directory Structure

```
llm/
├── modules/                          # Core functionality modules
│   ├── buffer.py                      # Event buffering and prioritization
│   ├── commentator.py                 # Natural speech output management
│   ├── event_cleaner.py               # Event data validation and processing
│   ├── llm_interface.py               # LLM model interaction
│   └── prompt_generator.py            # Context-aware prompt generation
├── tests/                            # Test suites
│   ├── resources/                     # Test data
│   ├── test_buffer.py                 # Buffer component tests
│   ├── test_llm_interface.py          # LLM interface tests
│   ├── test_pipeline.py               # Multi-threaded pipeline tests
│   ├── test_pipeline_presentation.py  # Presentation pipeline tests
│   └── test_prompt_generator.py       # Prompt generation tests
├── config.py                         # System configuration
└── llm.py                            # Main module entry point
```

## 🔧 Components

### Core Modules

1. **Buffer (`modules/buffer.py`)**
   - Priority-based event queueing
   - Overflow management
   - Interest score calculation

2. **Commentator (`modules/commentator.py`)**
   - Natural speech delivery
   - Interrupt handling
   - Speech pacing control

3. **Event Cleaner (`modules/event_cleaner.py`)**
   - Event data validation
   - Priority assignment
   - Interrupt detection

4. **LLM Interface (`modules/llm_interface.py`)**
   - Model initialization and management
   - Text generation
   - Error handling

5. **Prompt Generator (`modules/prompt_generator.py`)**
   - Context-aware prompt creation
   - Template management
   - Event type matching

## 🚀 Features

- Real-time event processing
- Natural language commentary
- Priority-based event handling
- Interrupt support for critical events
- Statistical insights generation
- Multi-threaded pipeline architecture

## 🧪 Testing

The `tests/` directory contains test suites for each component:

- Unit tests for individual modules
- Integration tests for the pipeline
- Performance tests for real-time processing
- Presentation-ready pipeline tests

## ⚙️ Configuration

System settings are managed through `config.py`, including:
- LLM model parameters
- Template configurations
- Buffer settings
- Speech parameters

## 🔑 Requirements

- Python 3.12.9
- HuggingFace Transformers
- Environment variable: `HUGGINGFACE_TOKEN` for model access

## 🚦 Pipeline Architecture

The system follows a producer-consumer architecture:
1. Event Producer: Processes and validates game events
2. Buffer: Manages event priority and queueing
3. Commentary Generator: Creates contextual prompts
4. Speech Output: Delivers natural commentary

## 🔍 Usage Example

```python
from llm.modules import llm_interface, commentator

# Initialize components
llm = llm_interface.LLMInterface(model="meta-llama/Llama-3.2-3B-Instruct")
speaker = commentator.Commentator()

# Process event and generate commentary
event = {"event": "goal", "team": "Team A", "player": "Player 1"}
comment = llm.generate_comment(event)
speaker.speak(comment)
```

## 🔗 Related Components

- `cv/` - Computer vision for event detection
- `markov/` - Statistical modeling
- `visual/` - Visualization components