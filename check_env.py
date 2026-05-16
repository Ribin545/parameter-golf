import torch, triton, sentencepiece, numpy, sys
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A")
print("Triton:", triton.__version__)
print("SentencePiece: OK")
print("NumPy:", numpy.__version__)