import torch
import sys

# Try to mock flashinfer if not in a CUDA environment
try:
    import flashinfer
    print("Found flashinfer!")
    if hasattr(flashinfer, 'mxfp8_quantize'):
        import inspect
        print("Signature:", inspect.signature(flashinfer.mxfp8_quantize))
        print("Docstring:", inspect.getdoc(flashinfer.mxfp8_quantize))
    else:
        print("mxfp8_quantize not found in flashinfer")
except ImportError as e:
    print(f"Import error: {e}")
