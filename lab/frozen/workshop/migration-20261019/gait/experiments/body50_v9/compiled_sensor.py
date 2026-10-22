"""Optional fixed-shape compilation around the frozen BODY50 producer.

No profile, numeric dtype, fusion initialization or public sensor packet field
changes. Noise stays outside the compiled function, using the same sensor RNG
and one[N,6] draw per physical tick as the eager producer. This module is an
experiment until exact packet and state checks pass on the target GPU.
"""
import torch


class CompiledBody50:
    def __init__(self, producer, *, backend='inductor', mode=None):
        if producer.sensor.strict_validation:
            raise ValueError('Compile only the private integer-clock producer with strict_validation=False; CPU validation remains a separate oracle')
        self.producer = producer
        self.original_update = producer.update
        self.backend, self.mode = backend, mode
        # Installed Inductor maps emulate_precision_casts to Triton
        # enable_fp_fusion=False, retaining eager intermediate roundings.
        options = {'emulate_precision_casts':True} if backend == 'inductor' and producer.device.type == 'cuda' else None
        self.compiled_update = torch.compile(self.original_update, backend=backend,
            mode=mode, options=options, fullgraph=True, dynamic=False)
        self.calls = 0

    def update(self, **kwargs):
        noise = kwargs.get('noise_standard_normal')
        if noise is None:
            sensor = self.producer.sensor
            noise = torch.randn((sensor.n,6), generator=sensor.generator,
                                device=sensor.device, dtype=torch.float64)
        kwargs['noise_standard_normal'] = noise
        result = self.compiled_update(**kwargs)
        self.calls += 1
        return result

    def report(self):
        return {'schema':'microduck.body50-compiled-producer.v1',
                'backend':self.backend,'mode':self.mode,'calls':self.calls,
                'fullgraph':True,'dynamic_shapes':False,'internal_dtype':'float64',
                'random_sampling':'one original-generator[N,6]draw outside compiled graph per physical tick',
                'emulate_precision_casts':True if self.backend=='inductor' and self.producer.device.type=='cuda' else 'not overridden',
                'frozen_sensor_source_modified':False,
                'physics_and_task_qualification_claimed':False}


def install(producer, **kwargs):
    adapter = CompiledBody50(producer, **kwargs)
    producer.update = adapter.update
    return adapter
