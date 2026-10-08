"""Prompt templates for the Synthesizer and the Reasoner."""
from .prompts import (
    DEFAULT_EMPTY_DIAGNOSTIC_REPORT,
    DEFAULT_EMPTY_POOL_REPORT,
    DEFAULT_EMPTY_REJECTION_SUMMARY,
    REASONER_SYSTEM_PROMPT,
    SYNTHESIZER_OUTPUT_SCHEMA_TEXT,
    SYNTHESIZER_REQUIRED_FIELDS,
    SYNTHESIZER_SYSTEM_PROMPT,
    SynthesizerInput,
    render_reasoner_prompt,
    render_synthesizer_prompt,
)

__all__ = [
    "DEFAULT_EMPTY_DIAGNOSTIC_REPORT",
    "DEFAULT_EMPTY_POOL_REPORT",
    "DEFAULT_EMPTY_REJECTION_SUMMARY",
    "REASONER_SYSTEM_PROMPT",
    "SYNTHESIZER_OUTPUT_SCHEMA_TEXT",
    "SYNTHESIZER_REQUIRED_FIELDS",
    "SYNTHESIZER_SYSTEM_PROMPT",
    "SynthesizerInput",
    "render_reasoner_prompt",
    "render_synthesizer_prompt",
]
