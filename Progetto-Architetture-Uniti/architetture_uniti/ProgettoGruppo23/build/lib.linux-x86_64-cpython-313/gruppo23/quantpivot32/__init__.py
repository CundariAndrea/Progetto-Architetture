"""
32-bit Quantized Pivot Indexing
"""

try:
    from ._quantpivot32 import QuantPivot
except ImportError:
    class QuantPivot:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "Errore: Non puoi istanziare la classe 32-bit direttamente da Python 64-bit. "
                "Il file test.py userà l'eseguibile esterno automaticamente."
            )

__all__ = ['QuantPivot']