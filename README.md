## 📘 Overview

Smart assistants and smart microphones can contribute to detecting verbal abuse by analyzing speech-to-text contents. In this paper, we compare different **large language models (LLMs)** for detecting verbal abuse and propose a framework that first identifies emotion from short audio conversations by extracting **Mel Frequency Cepstral Coefficient (MFCC) and Mel Spectrogram (MEL)** features. It then utilizes **transfer learning** with a fully connected neural network incorporating an attention mechanism and **SBERT** encoding. To demonstrate the efficacy of the proposed framework, we prepared a custom dataset containing instances of verbal abuse. Evaluation results show that our framework is lightweight and achieves commendable accuracy compared to the existing LLM models.

The repository includes:
- **`multi_label_saf.py`** — core model training and evaluation  
- **`k_folded_context_detection_saf.py`** — K-Fold cross-validation framework for performance reliability  
- **`Emotions_All_Data.csv`** — dataset containing multi-emotion annotations  
- **`family_conflict_conv.xlsx`** — conversational transcripts for contextual understanding  

---

## ⚙️ Features

- Multi-label classification (each utterance may express several emotions)
- Sentence-level semantic embeddings using **SBERT/BERT**
- Attention mechanism for contextual fusion
- K-Fold validation to ensure generalization
- Metrics: Precision, Recall, F1-score
- End-to-end preprocessing and tokenization pipeline
- Easily extendable for new emotion taxonomies or datasets

---

## 🧩 Project Structure

```
.
├── multi_label_saf.py                 # Main SAF training script
├── k_folded_context_detection_saf.py  # K-Fold validation script
├── family_conflict_conv.xlsx          # Conversation dataset
├── Emotions_All_Data.csv              # Emotion-labeled dataset
└── README.md
```

---

## 🧠 Model Architecture

The **Smart Annotation Framework (SAF)** model integrates:
1. **Sentence Embeddings:** obtained from Sentence-BERT or a similar transformer encoder  
2. **Attention Layer:** captures contextual dependencies between utterances  
3. **Dense Layers:** project contextualized embeddings into the multi-label output space  

```python
# Simplified example
encoded = SentenceTransformer('all-mpnet-base-v2').encode(sentences)
contextual = AttentionLayer()(encoded)
output = Dense(num_labels, activation='sigmoid')(contextual)
```

---

## 🧪 Datasets

| Dataset | Description | Format | Labels |
|----------|-------------|--------|---------|
| `Emotions_All_Data.csv` | Emotion dataset with multiple labels per utterance | CSV | Joy, Sadness, Anger, Fear, Disgust, Surprise |
| `family_conflict_conv.xlsx` | Multi-turn family conflict dialogues | XLSX | Contextual and emotional turns |

---

## 🚀 Usage

### 1️⃣ Installation

```bash
git clone https://github.com/<your-username>/multi-label-saf.git
cd multi-label-saf
pip install -r requirements.txt
```

### 2️⃣ Train the Model

```bash
python multi_label_saf.py
```

### 3️⃣ Run K-Fold Cross-Validation

```bash
python k_folded_context_detection_saf.py
```

### 4️⃣ Evaluate Results

Metrics are automatically saved and printed:
- Precision / Recall / F1
- Label-wise ROC curves
- K-Fold averaged performance

---

## 📊 Example Output

| Fold | Precision | Recall | F1-Score |
|------|------------|--------|----------|
| 1 | 0.84 | 0.82 | 0.83 | 
| 2 | 0.86 | 0.83 | 0.84 | 
| 3 | 0.85 | 0.84 | 0.85 | 

---

## 🧰 Requirements

- Python ≥ 3.8  
- TensorFlow / Keras ≥ 2.12  
- scikit-learn ≥ 1.3  
- pandas, numpy, matplotlib  
- sentence-transformers  

Install dependencies:

```bash
pip install tensorflow scikit-learn sentence-transformers pandas numpy matplotlib
```

---

## 🧩 Extending the Framework

To adapt the framework for a new dataset:
1. Place the dataset file (CSV/XLSX) in the project root.  
2. Update the `DATA_PATH` variable in `multi_label_saf.py`.  
3. Modify `LABEL_COLUMNS` to match your new annotation scheme.  
4. Run the training script — the pipeline automatically preprocesses and trains.

---

## 📖 Citation

If you use this dataset or code, please cite the following paper:

> F. A. Irfan, C. Behl and R. Iqbal, "Verbal Abuse Detection from Short Conversations," *2025 IEEE 22nd Consumer Communications & Networking Conference (CCNC)*, Las Vegas, NV, USA, 2025, pp. 1–2.  
> DOI: [https://doi.org/10.1109/CCNC54725.2025.10976181](https://doi.org/10.1109/CCNC54725.2025.10976181)

**BibTeX:**
```bibtex
@inproceedings{irfan2025verbal,
  author    = {F. A. Irfan and C. Behl and R. Iqbal},
  title     = {Verbal Abuse Detection from Short Conversations},
  booktitle = {2025 IEEE 22nd Consumer Communications \& Networking Conference (CCNC)},
  year      = {2025},
  pages     = {1--2},
  address   = {Las Vegas, NV, USA},
  doi       = {10.1109/CCNC54725.2025.10976181},
  keywords  = {Emotion recognition; Attention mechanisms; Accuracy; Transfer learning; Neural networks; Oral communication; Bidirectional control; Feature extraction; Encoding; Mel frequency cepstral coefficient; Attention mechanism; BERT; Emotion detection; K-fold validation; Neural network; SBERT; Transfer learning}
}
```

---


## 🪪 License

This project is released under the **MIT License**.  
See [LICENSE](LICENSE) for more details.

---

