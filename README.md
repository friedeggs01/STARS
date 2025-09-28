# Exploring Diverse Generation Paths via Inference-time Stiefel Activation Steering


We introduce **STAR** (**St**iefel-based **A**ctivation Steering for Diverse **R**easoning), a method to enhance diversity in reasoning processes during inference.

---

## 📂 Repository Structure

The repository is organized into two main tasks:

### 1. **TestEval** (Test Case Generation)

Located in the `testeval/` directory.

* `bestofn.py`: Baseline **temperature sampling**.
* `steering.py`: Proposed **STAR algorithm**.
* `data_utils.py`: Data loading and processing utilities.
* `eval_overall.py`: Evaluation scripts.
* `prompt_utils.py`: Prompt handling functions.
* `load_data.ipynb`: Notebook for preparing and exploring datasets.
* `croissant.json`: Metadata/configuration file.

### 2. **LiveIdeaBench** (Scientific Discovery)

Located in the `liveideabench/` directory.

* `bestofn.py`: Baseline **temperature sampling**.
* `steering.py`: Proposed **STAR algorithm**.
* `data/`, `keywords_data/`: Benchmark datasets.

---

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/STAR-ICLR26.git
cd STAR-ICLR26
pip install -r requirements.txt
```

---

## ▶️ Usage

### Running Baselines (Best-of-N Sampling)

```bash
python testeval/bestofn.py 
python liveideabench/bestofn.py 
```

### Running STAR (Our Proposed Algorithm)

```bash
python testeval/steering.py 
python liveideabench/steering.py 
```

---

## 📊 Tasks & Benchmarks

* **TestEval**: Test case generation benchmark.
* **LiveIdeaBench**: Scientific discovery benchmark.

Each benchmark evaluates the diversity and correctness of reasoning paths under baseline sampling and STAR steering.

---


