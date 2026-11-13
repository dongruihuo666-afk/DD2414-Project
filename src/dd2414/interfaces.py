"""Existing branch feature access; caller controls model/autograd/device policies."""
def prepare_inputs(batch, device):
    from dd2414.models.multisweep import prepare_bevcar_inputs as prepare
    return prepare(batch, device)

def extract_bev(model, inputs):
    return model.features(inputs, location='fused_bev')[0]
