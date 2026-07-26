"""CRG evaluation modules.

Kept import-free so that pulling in one module does not drag in the rest: `config`
and `data` are enough to validate the dataset or inspect results, without loading
torch, the CRG decode engine, or the TDDN encoder stack.
"""
