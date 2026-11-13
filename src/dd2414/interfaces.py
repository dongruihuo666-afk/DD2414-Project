"""Prepare model inputs and read fused BEV features.

The caller selects the device, train/eval mode and autograd context.
"""


def prepare_inputs(batch, device):
    from dd2414.models.multisweep import prepare_bevcar_inputs as prepare

    return prepare(batch, device)


def extract_bev(model, inputs):
    return model.features(inputs, location='fused_bev')[0]
