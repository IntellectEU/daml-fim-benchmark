"""Prompt construction for FIM evaluation.

Formats a benchmark example's prefix/suffix (and any provider-specific metadata)
into the prompt layout expected by each supported prompt type.
"""

from types import SimpleNamespace


def default_prompt_type(provider: str) -> str:
    """Return the default prompt type for the given provider.

    Args:
        provider: Inference provider name (e.g. ``"huggingface"`` or ``"openai"``).

    Returns:
        The prompt-type identifier used when no explicit type is configured.

    Raises:
        ValueError: If the provider has no known default prompt type.
    """
    if provider == "huggingface":
        return "PSM"
    if provider == "openai":
        return "CUSTOM_INSTR"
    raise ValueError(f"No default prompttype exists for provider: {provider}")


def resolved_prompt_type(config: SimpleNamespace) -> str:
    """Return the effective prompt type for a config.

    Uses ``config.prompt_type`` when set, otherwise falls back to the provider
    default from :func:`default_prompt_type`.

    Args:
        config: Configuration carrying ``prompt_type`` and ``provider``.

    Returns:
        The resolved prompt-type identifier.
    """
    prompt_type = config.prompt_type
    if prompt_type is None:
        prompt_type = default_prompt_type(config.provider)
    return prompt_type


def construct_prompt(row: dict, config: SimpleNamespace) -> str:
    """Build the model-ready prompt string for a single FIM example.

    Dispatches on the resolved prompt type to format the example's ``prefix``
    and ``suffix`` (and provider-specific metadata) into the layout expected by
    the target provider. Supported types are ``L2R``, ``PSM``, ``SPM``,
    ``IPF``, ``1S`` and ``CUSTOM_INSTR``.

    Args:
        row: Example fields, including ``prefix`` and ``suffix``.
        config: Configuration controlling provider, prompt type and
            provider-specific options.

    Returns:
        The formatted prompt string.

    Raises:
        ValueError: If the resolved prompt type is not recognized.
    """
    prompt_type = resolved_prompt_type(config)

    prefix = row["prefix"]
    suffix = row["suffix"]

    if prompt_type == "L2R":
        return prefix

    if prompt_type == "PSM":
        return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"

    if prompt_type == "SPM":
        return f"{suffix}\n[END]\n{prefix}"

    if prompt_type == "IPF":
        return f"{prefix}\n[MASK]\n{suffix}\n[END]\nComplete the masked part:\n\n{prefix}"

    if prompt_type == "1S":
        example = """Define a standard template
template Auto with owner : Party where
[MASK]
[END] signatory owner"""

        task = f"""{prefix}
[MASK]
{suffix}
[END]"""
        return f"{example}\n\n{task}"

    if prompt_type == "CUSTOM_INSTR":
        prefix_end = prefix[-10:].replace("\n", "\\n")
        suffix_start = suffix[:10].replace("\n", "\\n")

        return (
            "You are a code-infilling tool. Your task is to provide the missing string "
            "that bridges the gap between the provided PREFIX and SUFFIX.\n\n"
            "PREFIX:\n"
            f"{prefix}\n\n"
            "SUFFIX:\n"
            f"{suffix}\n\n"
            "INSTRUCTIONS:\n"
            "1. Output only the missing code wrapped in <FILL> tags.\n"
            f"2. The PREFIX ends with: '...{prefix_end}'. Start your output exactly after this sequence.\n"
            f"3. The SUFFIX starts with: '{suffix_start}...'. Stop your output exactly before this sequence.\n"
            "4. Do NOT repeat the prefix or suffix text inside the tags.\n\n"
            "Format: <FILL>code_here</FILL>"
        )

    raise ValueError(f"Prompt type not recognized: {prompt_type}")
