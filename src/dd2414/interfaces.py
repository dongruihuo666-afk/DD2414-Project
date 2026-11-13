"""Existing branch feature access; caller controls model/autograd/device policies."""
def prepare_inputs(batch, device):
    from dd2414.models.hybrid import prepare_inputs as prepare
    return prepare(batch, device)

def extract_bev(model, inputs):
    return model.shared_bev(*inputs)
