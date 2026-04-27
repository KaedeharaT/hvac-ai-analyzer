# HVAC AI Analyzer

A Python-based HVAC data analysis tool for automatic semantic column recognition, COP calculation, and system performance evaluation.

## Overview

HVAC AI Analyzer is designed to support building energy data analysis by combining rule-based logic, LLM-assisted semantic recognition, and physical consistency checks.

The tool can automatically identify key HVAC data columns such as supply temperature, return temperature, flow rate, power consumption, and energy-related variables. It also supports COP analysis, load ratio analysis, visualization, and report export.

## Key Features

- Automatic HVAC column semantic recognition
- C1–C8 multi-slot semantic scoring framework
- COP calculation using temperature, flow rate, and power data
- Load ratio analysis
- GUI-based operation with PyQt5
- Excel / Word report export
- Ground truth based evaluation module

## System Architecture

```text
Input Data
   ↓
Column Semantic Recognition
   ↓
Physical Consistency Check
   ↓
COP / Load Ratio Analysis
   ↓
Visualization and Report Export

## Workflow

```mermaid
flowchart TD
    A[Input BEMS / HVAC Data<br/>CSV or Excel] --> B[Data Loading & Preprocessing]

    B --> C[Column Evidence Extraction]
    C --> C1[Textual Evidence<br/>Column names and neighboring columns]
    C --> C2[Statistical Evidence<br/>zero ratio, monotonicity, variability, step change]
    C --> C3[Unit Evidence<br/>temperature, flow, power, energy, pressure]

    C1 --> D[Slot-based Semantic Representation]
    C2 --> D
    C3 --> D

    D --> E[C1-C8 Multi-slot Scoring]
    E --> F[LLM-based Hypothesis Generation]
    F --> G[Physics-aware Validation]

    G --> H{Reliable Interpretation?}

    H -->|Yes| I[Final Semantic Label]
    H -->|No| J[ABSTAIN]

    I --> K[HVAC Analysis]
    K --> K1[COP Calculation]
    K --> K2[Load Ratio Analysis]
    K --> K3[Visualization]

    K1 --> L[Report Export]
    K2 --> L
    K3 --> L

    L --> M[Excel / Word / Figures]
