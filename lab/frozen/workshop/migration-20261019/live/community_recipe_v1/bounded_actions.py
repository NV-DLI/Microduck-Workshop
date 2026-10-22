"""Experimental target limiter; never projects physical state into bounds.

Native Newton BAM compares its target with servo+passive joint position. The
returned policy action inverts the official scale/home/encoder-bias mapping.
Margins and slew limits are stated engineering assumptions, not proven safety.
"""
from pathlib import Path
import hashlib,json,math
import torch

JOINT_NAMES=['left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle','neck_pitch','head_pitch','head_yaw','head_roll','right_hip_yaw','right_hip_roll','right_hip_pitch','right_knee','right_ankle']

def load_limits(path=None):
    path=Path(path or Path(__file__).with_name('bounded-action-limits.json'))
    data=json.loads(path.read_text());source=Path(data['source_usd'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=data['source_usd_sha256']:raise ValueError('Named body-joint limit source changed')
    if data['joint_names']!=JOINT_NAMES:raise ValueError('Explicit14joint order required')
    return data

class BoundedActions:
    def __init__(self,limits,*,margin_rad=.035,max_target_rate_rad_s=3.,control_dt=.02):
        if not all(math.isfinite(v)and v>0 for v in (margin_rad,max_target_rate_rad_s,control_dt)):raise ValueError('Positive finite explicit limiter assumptions required')
        self.limits=limits;self.margin=margin_rad;self.rate=max_target_rate_rad_s;self.dt=control_dt;self.previous=None
    def reset(self):self.previous=None
    def apply(self,raw_action,*,scale,offset,encoder_bias,q_servo,q_passive,servo_limits):
        if raw_action.shape!=q_servo.shape or q_servo.shape!=q_passive.shape or raw_action.shape[-1]!=14:raise ValueError('Named policy/servo/passive arrays must have same[N,14]shape')
        if not all(torch.isfinite(x).all()for x in (raw_action,q_servo,q_passive,encoder_bias,servo_limits)):raise ValueError('Nonfinite physical/action input')
        scale=torch.as_tensor(scale,device=raw_action.device,dtype=raw_action.dtype);offset=torch.as_tensor(offset,device=raw_action.device,dtype=raw_action.dtype)
        if not torch.isfinite(scale).all()or not torch.isfinite(offset).all()or (scale<=0).any():raise ValueError('Finite positive action scale and finite home required')
        output_limits=torch.as_tensor(self.limits,device=raw_action.device,dtype=raw_action.dtype)
        # Each output limit is from the official non-backlash14joint asset,
        # never widened by adding the allowed play-hinge coordinate range.
        lower=output_limits[:,0]+self.margin;upper=output_limits[:,1]-self.margin
        lower=torch.maximum(lower,servo_limits[:,:,0]+self.margin+q_passive)
        upper=torch.minimum(upper,servo_limits[:,:,1]-self.margin+q_passive)
        if (lower>=upper).any():raise ValueError('No admissible output/servo target intersection')
        requested=raw_action*scale+offset-encoder_bias
        clamped=torch.maximum(lower,torch.minimum(upper,requested))
        previous=q_servo+q_passive if self.previous is None else self.previous
        rate_limited=torch.maximum(previous-self.rate*self.dt,torch.minimum(previous+self.rate*self.dt,clamped))
        # A moved admissible interval can conflict with the slew bound. Prefer
        # the explicit output boundary and record that exception; do not hide it.
        target=torch.maximum(lower,torch.minimum(upper,rate_limited))
        result=(target+encoder_bias-offset)/scale
        diagnostics={'raw_policy_action':raw_action.clone(),'requested_output_target_rad':requested.clone(),'executed_output_target_rad':target.clone(),
            'boundary_limited':(clamped-requested).abs()>1e-7,'rate_limited':(rate_limited-clamped).abs()>1e-7,'boundary_overrode_rate':(target-rate_limited).abs()>1e-7,
            'measured_effective_output_rad':(q_servo+q_passive).clone(),'physical_state_modified':False}
        self.previous=target.detach().clone()
        return result,diagnostics
