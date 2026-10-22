"""Public 8x8 angular-zone association of a fresh observed sphere return.

This explains only a nearest return; it never marks the rest of a zone free.
It is a development measurement model, not an ST firmware/noise calibration.
"""
import math
import numpy as np


def nearest_sphere_range(center, radius, row, col, hfov=45., vfov=45.):
    """Exact nearest sphere surface across the continuous spherical-angle cell."""
    c=np.asarray(center,dtype=float)
    if c.shape!=(3,) or not np.isfinite(c).all() or np.linalg.norm(c)<=radius:
        return None
    azlo,azhi=np.deg2rad([hfov*(.5-(col+1)/8),hfov*(.5-col/8)])
    ello,elhi=np.deg2rad([vfov*(.5-(row+1)/8),vfov*(.5-row/8)])
    az=float(np.clip(math.atan2(c[1],c[0]),azlo,azhi))
    horizontal=c[0]*math.cos(az)+c[1]*math.sin(az)
    el=float(np.clip(math.atan2(c[2],horizontal),ello,elhi))
    ray=np.array([math.cos(el)*math.cos(az),math.cos(el)*math.sin(az),math.sin(el)])
    along=float(c@ray);discriminant=along*along-(float(c@c)-radius*radius)
    if along<=0 or discriminant<0:return None
    return along-math.sqrt(discriminant)


def associate(cell, *, center_tof, camera_to_ball_unit_tof, range_uncertainty_m,
              timestamp_s, ball_timestamp_s, hfov=45., vfov=45., radius=.035):
    """Existing RGB size allowance propagates along its measured viewing ray.

    No world state, high-resolution depth, raster indices, or image-invisible
    ball extent is accepted. Unknown lens/radius/FK errors are not silently
    added to the allowance. A bounded65-point sweep may conservatively reject
    feasible intermediate centres; it does not widen the angular cell.
    """
    result={'associated':False,'reason':None,'sphere_radius_m':radius,
            'semantics':'nearest return only; surrounding zone remains unknown',
            'truth_used':False,'hardware_calibrated':False}
    def reject(reason):result['reason']=reason;return result
    if abs(timestamp_s-ball_timestamp_s)>1e-6:return reject('fresh_same_capture_ball_required')
    if cell.get('contains_too_close') or cell.get('status')!='valid' or cell.get('range_m') is None:
        return reject('invalid_or_too_close')
    if cell.get('mixed_edge') or cell.get('invalid_samples',0) or cell.get('no_return_samples',0):
        return reject('mixed_or_unknown_samples')
    uncertainty=float(range_uncertainty_m)
    if not math.isfinite(uncertainty) or not 0<=uncertainty<=.025:
        return reject('visual_range_uncertainty_exceeds_25mm_budget')
    center=np.asarray(center_tof,float);direction=np.asarray(camera_to_ball_unit_tof,float)
    if center.shape!=(3,) or direction.shape!=(3,) or not np.isfinite(center).all() or not np.isfinite(direction).all():
        return reject('nonfinite_public_geometry')
    if abs(float(np.linalg.norm(direction))-1)>1e-6:return reject('view_ray_not_unit')
    ranges=[nearest_sphere_range(center+d*direction,radius,int(cell['row']),int(cell['col']),hfov,vfov)
            for d in np.linspace(-uncertainty,uncertainty,65)]
    ranges=[v for v in ranges if v is not None]
    result.update(range_uncertainty_m=uncertainty,uncertainty_source='existing RGB box extent allowance along camera ray',
                  quantization_allowance_m=.0005,angular_zone_expansion_deg=0.,uncertainty_center_samples=65)
    if not ranges:return reject('observed_sphere_uncertainty_misses_zone')
    lower=min(ranges)-.0005;upper=max(ranges)+.0005
    result.update(predicted_nearest_range_interval_m=[lower,upper],observed_range_m=float(cell['range_m']))
    if not lower<=float(cell['range_m'])<=upper:return reject('nearest_return_outside_sphere_interval')
    result.update(associated=True,reason='fresh_RGB_sphere_and_angular_zone_range_consistent')
    return result
