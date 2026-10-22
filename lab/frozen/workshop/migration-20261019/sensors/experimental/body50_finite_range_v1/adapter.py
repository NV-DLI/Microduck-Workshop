"""Explicit opt-in for new experiments only; never changes frozen producer files."""
from finite_range_core import VectorBodyImu


def install_before_first_tick(producer):
    if bool((producer.ticks != 0).any()):
        raise ValueError('Select the sensor hypothesis before any physical sensor tick')
    old = producer.sensor
    sensor = VectorBodyImu(old.n, device=old.device, profile=old.profile,
                           seed=0, strict_validation=old.strict_validation)
    sensor.generator.set_state(old.generator.get_state())
    producer.sensor = sensor
    return sensor
