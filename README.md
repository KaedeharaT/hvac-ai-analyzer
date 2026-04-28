# HVAC AI Analyzer

A Python-based intelligent HVAC data analysis system integrating **semantic column recognition, LLM-assisted reasoning, and physics-aware validation** for building energy performance analysis.

---

## 🎯 Overview

HVAC AI Analyzer is designed to automate the interpretation of building operation data (BEMS), reducing manual preprocessing effort and improving reliability.

The system combines:

- Rule-based logic
- LLM-assisted semantic reasoning
- Physics-aware validation

to achieve robust and interpretable understanding of HVAC datasets.

---

## 🧠 Key Contributions

- 🔹 **Semantic–physical hybrid reasoning framework**
- 🔹 **C1–C8 multi-slot representation for HVAC variables**
- 🔹 **LLM-based hypothesis generation with constraint validation**
- 🔹 **Automatic COP calculation and energy performance analysis**
- 🔹 **ABSTAIN mechanism for uncertainty handling**

---

## 🏗 System Architecture

<p align="center">
  <a href="overview.pdf">
    <img src="overview.png" width="750"/>
  </a>
</p>

<p align="center">
  <em>Click the figure to view high-resolution PDF version.</em>
</p>

---

## ⚙️ Workflow
Input Data → Semantic Recognition → Physical Validation → HVAC Analysis → Report Output

---

## 📊 Features

- Automatic HVAC column semantic recognition
- Multi-source evidence extraction (textual / statistical / unit)
- LLM-based semantic interpretation
- Physics-aware consistency validation
- COP and load ratio analysis
- Visualization and report generation
- GUI support (PyQt5)

---

## 🧪 Example Applications

- Building energy performance analysis
- HVAC system monitoring
- Data cleaning and semantic alignment for BEMS datasets
- AI-assisted engineering analysis

---

## 🛠 Tech Stack

- Python
- Pandas / NumPy
- PyQt5 (GUI)
- Matplotlib (Visualization)
- LLM API (semantic reasoning)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/KaedeharaT/hvac-ai-analyzer.git
cd hvac-ai-analyzer
2. Install dependencies
pip install pandas numpy matplotlib pyqt5 openpyxl python-docx
3. Run the system
python main.py
4. Usage
Launch the GUI
Load your HVAC data (CSV / Excel)
Select analysis mode (COP / Load Ratio / Column Recognition)
Run analysis
Export results (Excel / Word / figures)



