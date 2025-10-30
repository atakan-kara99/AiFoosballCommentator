# A Live Foosball Commentary System Using Computer Vision, Markov Models, and Large Language Models 

### *Computer Vision • Markov Models • Large Language Models*

**Authors:** Christoph Crasemann, Julius Daum, Carolina Horlbeck, Atakan Kara, Joost Krüger, Falk Kujath, Luca Mayer, Mads Nissen, Jonathan Schirp, Philipp Wieck

**Date:** March 31, 2025

---

## 🎯 Overview

Sports commentary enriches modern sports entertainment by providing **real-time insights, narrative context, and expert analysis**.

While professional sports benefit from automated data-driven commentary, **foosball has seen limited innovation** in this space. Despite existing tracking tools, **no prior work has targeted autonomous, real-time commentary generation for foosball** — until now.

This project presents a **complete pipeline** for generating live foosball commentary using:

* 🎥 *Computer Vision*
* 🔁 *Markov State Modeling*
* 🧠 *Large Language Models (LLMs)*
* 🗣️ *Text-to-Speech playback*

---

## 🧩 System Architecture

### 1. **Digital Twin via Computer Vision**

A live digital twin of the foosball table is constructed from video input, detecting:

* Player rod positions
* Ball movement
* Game events

> U-Net models are used for precise player and ball segmentation.

---

### 2. **Game State Modeling**

A **Markov model** extracts structured, interpretable game insights by identifying:

* Possession patterns
* Offensive vs. defensive play
* Event transitions (passes, shots, goals)

---

### 3. **Autonomous Commentary Generation**

The extracted events are fed into an **LLM**, which outputs:

* Natural-language narration
* Game highlights
* Tactical and strategic context

---

### 4. **Real-Time Delivery**

To enhance playback and usability:

* 🖥️ Custom video player for analysis
* 📊 Frontend with key performance indicators (KPIs)
* 🗣️ Browser-based text-to-speech for live audio commentary
* 🐳 Docker-based deployment for portability and reproducibility

---

## ⚡ Performance Focus

This system is built with **low-latency, near-real-time commentary** as a core requirement.

Key design priorities:

* Minimal delay in computer-vision inference
* Efficient Markov model updates
* Fast natural-language generation
* Smooth UI streaming and audio playback

---

## 🚀 Contributions

| Component       | Contribution                          |
| --------------- | ------------------------------------- |
| Computer Vision | U-Net-based player and ball tracking  |
| State Modeling  | Real-time Markov event extraction     |
| LLM Pipeline    | Automated play-by-play commentary     |
| Frontend        | KPI dashboard + TTS playback          |
| Deployment      | Modular Docker-based system           |
| Tooling         | Custom video review / annotation tool |

---

## 🏁 Summary

This work demonstrates a **first-of-its-kind autonomous commentary system for foosball**, combining real-time computer vision, probabilistic modeling, and AI-driven language generation.

It represents an early step toward **AI-augmented sports broadcasting**, offering exciting opportunities for:

* Small-scale sports
* Event automation
* Real-time analytics and fan engagement
