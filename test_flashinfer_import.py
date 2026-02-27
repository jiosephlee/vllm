import sys
# Try loading flashinfer directly from the site-packages if we can find it
import importlib.util

try:
    # First see if it's available via normal import
    import flashinfer
    print("Direct import successful")
    print("Has mxfp8_quantize?", hasattr(flashinfer, "mxfp8_quantize"))
except ImportError:
    print("Could not import flashinfer directly")

import site
print("Site packages:", site.getsitepackages())
