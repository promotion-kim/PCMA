import importlib.util
import os


def is_peft_available():
    return importlib.util.find_spec("peft") is not None


def is_wandb_available():
    if importlib.util.find_spec("wandb") is None:
        return False

    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return False

    if os.environ.get("WANDB_DISABLED", "").lower() in {"true", "1", "yes"}:
        return False

    return True
