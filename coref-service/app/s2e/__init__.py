"""Vendored subset of yuvalkirstain/s2e-coref (ACL 2021).

"Coreference Resolution without Span Representations", Kirstain, Ram & Levy.
Source: https://github.com/yuvalkirstain/s2e-coref (MIT License).

Only the model definition and the helpers required for *inference* are vendored
here; the training/eval scripts are not. The imports in `modeling.py` were
patched to run under the transformers version pinned by this service.
"""
