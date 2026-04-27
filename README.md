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
