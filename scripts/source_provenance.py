"""Content-verified source relocation; never used to relax exact resume."""


def check_camera_source_provenance(config, checkpoint_path, initialization, allow_relocation=False):
    expected = str(checkpoint_path.resolve()) if checkpoint_path else None
    original = config.get('official_camera_checkpoint')
    stored = config.get('official_camera_initialization')
    if not allow_relocation:
        if original != expected:
            raise ValueError('source official encoder provenance mismatch')
        if initialization and stored != initialization:
            raise ValueError('source official encoder hash mismatch')
        return
    if not checkpoint_path or not initialization or not stored:
        raise ValueError('Relocation requires both explicit official encoder records')
    # Only the location changes. Hash, tensor count, and encoder-only scope must
    # still agree, and model.load_state_dict(strict=True) remains unchanged.
    if {k:v for k,v in stored.items() if k != 'path'} != {
            k:v for k,v in initialization.items() if k != 'path'}:
        raise ValueError('source official encoder content/provenance mismatch')
