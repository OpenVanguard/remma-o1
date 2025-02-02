### **Usage**

1. **Build Image**:
```bash
docker build -t remma-llm .
```

2. **Run with Jupyter**:
```bash
docker run -it --gpus all \
  -p 8888:8888 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  remma-llm
```

3. **Run Training**:
```bash
docker run -it --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  remma-llm \
  python -m src.training.train_custom
```
