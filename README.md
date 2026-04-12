# Drug-Target Binding Affinity Prediction using GNN

Predicts the binding affinity (Kd in nM) between drug molecules and protein targets
using a Graph Neural Network (GCN) + 1D CNN dual-branch architecture.

**Course:** EGN6217 — Engineering Applications of Machine Learning
**Semester:** Spring 2026 | University of Florida
**Author:** Sathyadharini Srinivasan | sathyadharini@ufl.edu

---

## Project Overview

Drug discovery takes 12+ years and $2.6B per drug on average. A key bottleneck is predicting
how strongly a drug molecule binds to a protein target (binding affinity, Kd in nM). This project
builds an end-to-end deep learning system: a Graph Neural Network processes the drug's molecular
graph, a 1D CNN encodes the protein sequence, and a fusion MLP predicts the binding affinity.

---

## Architecture

```
Drug SMILES ──► RDKit ──► Molecular Graph ──► GCNConv×3 ──► GlobalMeanPool ──► 128-dim ─┐
                                                                                           ├──► Concat(224) ──► MLP ──► Kd (nM)
Protein Sequence ──────────────────────► Embed ──► Conv1D×3 ──► GlobalMaxPool ──► 96-dim ─┘
```

| Component | Architecture | Output |
|-----------|-------------|--------|
| Drug Encoder | GCNConv(5→64→128) + BatchNorm + ReLU, GlobalMeanPool | 128-dim |
| Protein Encoder | Embedding(25,128) + Conv1D×3, GlobalMaxPool | 96-dim |
| Fusion MLP | FC(224→512)→FC(512→256)→FC(256→1) + Dropout(0.2) | Kd scalar |

See `docs/architecture_diagram.png` for the full system diagram.

---

## Dataset

**DeepDTA Davis Dataset**
- 442 unique drug compounds (SMILES format)
- 68 human kinase protein targets (amino acid sequences)
- 30,056 drug-target pairs with measured Kd binding affinity values
- Train / Test split: 25,046 train / 5,010 test (official DeepDTA fold protocol)

---

## Results (Deliverable 2 — Baseline, 10 epochs, CPU)

| Metric | Score |
|--------|-------|
| Test MSE (pKd) | 0.4793 |
| RMSE (pKd) | 0.6923 |
| Pearson r | 0.6061 |
| Concordance Index (CI) | 0.8011 |

See `results/training_results.png` for loss curves and predicted vs actual scatter plot.

---

## Project Structure

```
drug-target-binding-gnn/
├── data/davis/              ← Davis dataset (ligands, proteins, affinity matrix, folds)
├── docs/                    ← Architecture diagram, UI screenshots
├── notebooks/               ← Additional notebooks
├── src/
│   ├── graph_utils.py       ← SMILES → molecular graph, protein sequence encoding
│   └── model.py             ← DTAModel: DrugEncoder + ProteinEncoder + MLP regressor
├── ui/
│   └── app.py               ← Gradio web interface
├── results/
│   ├── dta_model.pt         ← Trained model checkpoint
│   ├── eda_plots.png        ← Exploratory data analysis (Week 1)
│   └── training_results.png ← Loss curves + predicted vs actual (Week 2)
├── setup.ipynb              ← Main notebook: EDA + training + evaluation (fully executed)
├── requirements.txt
└── README.md
```

---

## Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/drug-target-binding-gnn
cd drug-target-binding-gnn

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Install PyTorch Geometric
pip install torch-geometric

# 4. Install Gradio (for the UI)
pip install gradio
```

---

## How to Run

### Option 1 — Main Notebook (Training + Evaluation)

Open `setup.ipynb` in Jupyter or Google Colab and run all cells in order.

The notebook will:
1. Download the Davis dataset automatically
2. Validate all SMILES strings with RDKit
3. Generate EDA visualisations
4. Build the DTADataset and DataLoaders
5. Train the DTAModel for 10 epochs
6. Evaluate with MSE, RMSE, Pearson r, and Concordance Index
7. Save loss curves and predicted vs actual plots to `results/`

### Option 2 — Gradio Web Interface

```bash
cd ui
python app.py
# Open http://127.0.0.1:7860 in your browser
```

**Input:** Drug SMILES string + Protein amino acid sequence (max 1000 chars)
**Output:** Predicted Kd (nM), pKd value, colour-coded binding strength (Strong / Moderate / Weak), and 2D molecule visualisation

---

## Known Issues / Current Limitations

- Trained for only 10 epochs on CPU — more epochs and GPU will improve scores significantly
- Model is biased toward kinase inhibitors (Davis dataset scope)
- CI computed on a 2,000-pair sample for speed; full O(n²) computation is slow
- Week 3 will add hyperparameter tuning and GCN vs GIN comparison

---

## Contact

Sathyadharini Srinivasan
University of Florida — M.S. Artificial Intelligence
Email: sathyadharini@ufl.edu
