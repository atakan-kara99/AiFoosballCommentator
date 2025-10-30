# Foosball Commentator - Computer Vision Analysis

## Overview

This module is responsible for detecting and tracking key elements in a foosball game using computer vision techniques. It processes video frames to detect the ball, players, goals, and game events such as throw-ins and touches. The extracted data is then used to log game events and by that provide real-time analysis.

## Directory Structure

```
.
│   __init__.py
│   .gitignore
│   foosball_enums.py
│   global_constants.py
│   main.py
│   utils.py
│   touchlog.py
│
├───ball
│   │   __init__.py
│   │   ball_detection.py
│   │   color_detection.py
│   │   goal_detection.py
│   │   hough_transform.py
│   │   main.py
│   │   ball_approx.py
│   │   throw_in_detection.py
│
├───debug_player
│   │   __init__.py
│   │   README.md
│   │   debug_overlay.py
│   │   user_input_handler.py
│   │   base_video_processor.py
│   │   decord_video_processor.py
│   │   opencv_video_processor.py
│
├───tests
│   │   __init__.py
│   │   evaluate.py
│   │   detect.py
│   │   test_touches.py
│   │
│   ├───annotated
│   │   ...
│   ├───detected
│   │   ...
│
├───resources
│       test_011.mp4
│       test_011_Tor2.mp4
│       test_011_2Tore.mp4
│
├───entities
│   │   __init__.py
│   │   entity.py
│   │   gamestate.py
│   │   player.py
│   │   zone.py
│   │   ball.py
│
├───field_detection
│   │   __init__.py
│   │   field_detection.py
│
└───player_detection
    │   aggregator.py
    │   ai.ipynb
    │   frame_exporter.py
    │   labeling_script.py
    │   long_exp_generator.py
    │   update_labels.py
    │   __init__.py
    │   player.py
    │   player_unet.py
    │   player_unet_debugger.py
    │   rod.py
    │   unet_foosball.pth
    │   util.py
    ├───training_images
```

## Functional Components

### 1. **Field Detection**

- Detects and crops the relevant part of the frame where the game is played.
- Located in `field_detection/field_detection.py`.

### 2. **Player Detection**

- Identifies and tracks player positions.
- Uses deep learning (`player_unet.py`) for player segmentation.
- Located in `player_detection/`.

### 3. **Ball Detection**

- Identifies the foosball in each frame.
- Uses color detection (`color_detection.py`), and ball approximation (`ball_approx.py`)
- **Goal Detection** (`goal_detection.py`)
- **Throw-in Detection** (`throw_in_detection.py`)
- **Touch Logging** (`touchlog.py`)
- Located in `ball/`.

### 4. **Gamestate**

- Maintains an up-to-date representation of the game.
- Stores ball position, player states, and game timestamps.
- Located in `entities/gamestate.py`.

### 5. **Touchlogger**

The `touchlog.py` module is responsible for logging ball interactions, tracking which player touched the ball, and determining key game events. It integrates information from ball detection, player detection from the gamestate to generate structured logs.

**Key Functions:**

- **Detect touches:** Determines if the ball changed its movement and which player last touched the ball based on proximity and movement direction.
- **Log game events:** Records touches, throw-ins, and goals along with their timestamps and player details.
- **Error handling:** Identifies inconsistencies in detected touches and logs them for debugging.

The touchlogger plays a crucial role in understanding the game flow and is used to enhance real-time commentary and analytics.

### 6. Evaluation
In `tests/` you find the data and scripts for evaluating goal, throw-in and touch detection.