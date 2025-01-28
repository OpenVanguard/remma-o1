### **Core Requirements**
| Library | Purpose | Installation |
|---------|---------|--------------|
| `torch` | Core deep learning framework | `pip install torch` |
| `transformers` | Pre-trained models & tokenizers | `pip install transformers` |
| `datasets` | Dataset loading/processing | `pip install datasets` |
| `numpy` | Numerical operations | `pip install numpy` |
| `tqdm` | Progress bars | `pip install tqdm` |

---

### **Data Processing**
| Library | Purpose |
|---------|---------|
| `spaCy`/`nltk` | Text preprocessing |
| `langdetect` | Language detection |
| `trafilatura` | HTML cleaning |
| `datasketch` | Deduplication |
| `sentencepiece` | Tokenization |
| `protobuf` | Serialization (for tokenizers) |
| `pyarrow` | Parquet file handling |

---

### **Modeling & Training**
| Library | Purpose |
|---------|---------|
| `accelerate` | Distributed training |
| `deepspeed` | Optimized training |
| `flash-attn` | Efficient attention |
| `bitsandbytes` | 4/8-bit quantization |
| `peft` | Parameter-efficient FT |

---

### **Evaluation**
| Library | Purpose |
|---------|---------|
| `evaluate` | Hugging Face metrics |
| `rouge-score` | Text generation metrics |
| `scikit-learn` | Classification metrics |

---

### **Utilities**
| Library | Purpose |
|---------|---------|
| `pyyaml` | Config files |
| `loguru` | Logging |
| `dvc` | Data versioning |
| `huggingface-hub` | Model sharing |

---

### **Optional (Advanced Use Cases)**
```bash
# Quantization/optimization
pip install flash-attn onnxruntime optimum

# Deployment
pip install fastapi uvicorn docker

# Alternative tokenizers
pip install tokenizers fastBPE

# CUDA 11.x specific (check compatibility)
pip install torch --extra-index-url https://download.pytorch.org/whl/cu117
```
