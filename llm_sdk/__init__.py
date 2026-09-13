"""Mock llm_sdk package -- stand-in for the real one during local testing.

See small_llm_model.py's module docstring for exactly what this does and
does not simulate faithfully.
"""

from llm_sdk.small_llm_model import Small_LLM_Model

__all__ = ["Small_LLM_Model"]
