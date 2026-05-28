"""Factory for instantiating an LLM provider from a string spec.

Accepts ``"<provider>:<model_id>"`` (e.g. ``"openai:gpt-4o"``,
``"anthropic:claude-3-5-sonnet-20240620"``) or a pre-built ``Model``
instance. Returns ``None`` for ``None``. Only the providers OpenAgent
ships drivers for are dispatched — anything else raises so a typo in a
config doesn't silently fall back to a wrong vendor.
"""

from typing import Optional, Union

from src.models.providers.base import Model


_PROVIDER_LOADERS: dict[str, tuple[str, str]] = {
    "anthropic": ("src.models.providers.anthropic", "Claude"),
    "openai": ("src.models.providers.openai", "OpenAIChat"),
    "google": ("src.models.providers.google", "Gemini"),
    "gemini": ("src.models.providers.google", "Gemini"),
    "deepseek": ("src.models.providers.deepseek", "DeepSeek"),
    "groq": ("src.models.providers.groq", "Groq"),
}


def _get_model_class(model_id: str, model_provider: str) -> Model:
    spec = _PROVIDER_LOADERS.get(model_provider)
    if spec is None:
        raise ValueError(
            f"Model provider '{model_provider}' is not supported. "
            f"Available providers: {sorted(_PROVIDER_LOADERS)}."
        )
    module_name, class_name = spec
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)
    return cls(id=model_id)


def _parse_model_string(model_string: str) -> Model:
    if not model_string or not isinstance(model_string, str):
        raise ValueError(f"Model string must be a non-empty string, got: {model_string}")

    if ":" not in model_string:
        raise ValueError(
            f"Invalid model string format: '{model_string}'. "
            "Model strings should be in format '<provider>:<model_id>' "
            "e.g. 'openai:gpt-4o'"
        )

    model_provider, model_id = model_string.split(":", 1)
    model_provider = model_provider.strip().lower()
    model_id = model_id.strip()

    if not model_provider or not model_id:
        raise ValueError(
            f"Invalid model string format: '{model_string}'. "
            "Model strings should be in format '<provider>:<model_id>' "
            "e.g. 'openai:gpt-4o'"
        )

    return _get_model_class(model_id, model_provider)


def get_model(model: Union[Model, str, None]) -> Optional[Model]:
    if model is None:
        return None
    elif isinstance(model, Model):
        return model
    elif isinstance(model, str):
        return _parse_model_string(model)
    else:
        raise ValueError("Model must be a Model instance, string, or None")
