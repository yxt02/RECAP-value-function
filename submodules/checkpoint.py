"""Validate the distributional checkpoint, including the earlier mislabeled format."""
import warnings

ARCHITECTURE = 'siglip_mean_patch_categorical'


def distribution_bins(payload):
    if payload.get('format_version') != 2 or payload.get('architecture') not in (
            ARCHITECTURE, 'siglip_mean_patch_scalar'):
        raise ValueError('Unsupported checkpoint format/architecture')
    state = payload['model_state_dict']
    centers = state.get('bin_centers')
    weight = state.get('value_head.2.weight')
    if centers is None or centers.ndim != 1 or len(centers) < 2 or weight is None or weight.shape[0] != len(centers):
        raise ValueError('Expected a categorical checkpoint; legacy scalar weights are incompatible')
    bins = len(centers)
    if payload['config'].get('num_bins', bins) != bins:
        raise ValueError('Checkpoint num_bins disagrees with weights')
    if payload['architecture'] != ARCHITECTURE:
        warnings.warn('Legacy architecture label says scalar; verified categorical weights are used.', stacklevel=2)
    return bins
