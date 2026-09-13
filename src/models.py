"""Pydantic data models for the function-calling schema and test inputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ParamType = Literal["number", "string", "boolean"]


class ParameterSpec(BaseModel):
    """Describes the expected type of a single function parameter."""

    model_config = ConfigDict(extra="forbid")

    type: ParamType = Field(..., description="Expected JSON type of the parameter.")


class ReturnSpec(BaseModel):
    """Describes the return type of a function."""

    model_config = ConfigDict(extra="forbid")

    type: ParamType = Field(..., description="Expected JSON type of the return value.")


class FunctionDefinition(BaseModel):
    """A single callable function exposed to the LLM."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: ReturnSpec


class TestPrompt(BaseModel):
    """A single natural-language prompt to be resolved into a function call."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1)


class FunctionCallResult(BaseModel):
    """A single resolved function call, ready to be written to the output file."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, object]
