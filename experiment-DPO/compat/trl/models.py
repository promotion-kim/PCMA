import copy


class PreTrainedModelWrapper:
    """
    Minimal compatibility shim for old TRL imports.
    The custom trainer usually only needs this symbol for isinstance/type checks.
    """
    pass


def create_reference_model(model, num_shared_layers=None, pattern=None):
    """
    Minimal replacement for TRL's create_reference_model.

    In our DPO script, ref_model is already passed explicitly, so this is usually
    not called. If it is called, we deepcopy and freeze the model.
    """
    ref_model = copy.deepcopy(model)
    ref_model.eval()

    for param in ref_model.parameters():
        param.requires_grad = False

    return ref_model
