"""The policy always sees 14 encoders, independently of USD joint ordering."""
from contract import POLICY_JOINTS

VARIANTS = ('servo14', 'backlash28')
PASSIVE_JOINTS = tuple(f'passive_{name}_backlash' for name in POLICY_JOINTS)


def resolve_variant(explicit, runtime, *, mode, source, has_checkpoint=False):
    if explicit is not None:
        selected, reason = explicit, 'explicit CLI selection'
    elif runtime.get('robot_variant'):
        selected, reason = runtime['robot_variant'], 'checkpoint/runtime lineage'
    elif source == 'community' or has_checkpoint:
        selected, reason = 'servo14', 'community or historical checkpoint without backlash metadata'
    else:
        selected, reason = 'backlash28', 'new training/diagnostic default'
    if selected not in VARIANTS:
        raise ValueError(f'Unknown robot variant: {selected}')
    return selected, reason


def resolve_joint_channels(names, variant):
    """Reject ambiguous models rather than silently reordering actor channels."""
    if variant not in VARIANTS:
        raise ValueError(f'Unknown robot variant: {variant}')
    expected = POLICY_JOINTS + (PASSIVE_JOINTS if variant == 'backlash28' else ())
    if len(names) != len(expected) or len(set(names)) != len(names) or set(names) != set(expected):
        raise ValueError(f'{variant} requires exact named joints; found {names}')
    return ([names.index(name) for name in POLICY_JOINTS],
            [names.index(name) for name in PASSIVE_JOINTS] if variant == 'backlash28' else [])


def encoder_channels(values, servo_ids, passive_ids):
    """Physical output encoder = motor angle + series gearbox play angle.

    Applies equally to joint position and velocity; caller adds sensor bias and
    subtracts the 14-channel home pose only for the position observation.
    """
    measured = values.index_select(1, servo_ids)
    if passive_ids.numel():
        measured = measured + values.index_select(1, passive_ids)
    return measured
