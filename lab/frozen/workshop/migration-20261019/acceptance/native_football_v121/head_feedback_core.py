"""Bounded same-read outer integral correction to native policy head commands."""
import math
import numpy as np

HOME=np.array([.3491,.3491,0.,0.])
BOUND=np.array([1.1,1.1,1.4,.31])


class HeadSetpointFeedback:
    def __init__(self, n):
        self.bias=np.zeros((n,4));self.sent=np.zeros((n,4));self.last_seq=np.full(n,-1,np.int64)
        self.last_read=np.full(n,np.nan);self.last_now=None

    def update(self, *, now_s, nominal, measured_q, seq, read_s, valid, enabled, slew_rad_s=.3):
        if not math.isfinite(slew_rad_s) or not .3<=slew_rad_s<=.65:raise ValueError('Explicit bounded head slew required')
        n=len(self.sent);nominal=np.asarray(nominal,float);q=np.asarray(measured_q,float)
        seq=np.asarray(seq,np.int64);read=np.asarray(read_s,float);valid=np.asarray(valid,bool);enabled=np.asarray(enabled,bool)
        if not math.isfinite(now_s) or now_s<0 or self.last_now is not None and now_s<=self.last_now:
            raise ValueError('Finite increasing policy clock required')
        if nominal.shape!=(n,4) or q.shape!=(n,4) or any(x.shape!=(n,)for x in (seq,read,valid,enabled)):
            raise ValueError('Explicit per-world public input shape required')
        if not np.isfinite(nominal).all() or not np.isfinite(q).all() or np.any(abs(nominal)>BOUND+1e-9):
            raise ValueError('Finite nominal head target inside source command domain required')
        if np.any(valid & (~np.isfinite(read)|(read<0)|(read>now_s+1e-9)|(now_s-read>.08))):
            raise ValueError('Valid read must be fresh and causal')
        if np.any(valid & (self.last_seq>=0) & ((seq<self.last_seq)|(read<self.last_read))):
            raise ValueError('Monotone public sequence and read time required')
        if np.any(valid & (seq==self.last_seq) & (read!=self.last_read)):
            raise ValueError('Duplicate sequence must retain its original read clock')
        dt=.02 if self.last_now is None else now_s-self.last_now
        if dt>.025:raise ValueError('Policy clock gap requires caller restart')
        self.last_now=now_s
        error=nominal+HOME-q;delta=np.zeros_like(self.bias);used=np.zeros(n,bool)
        integral_error=error.copy();redistributed=np.zeros(n,bool)
        for i in range(n):
            if not valid[i]:
                self.last_seq[i]=-1;self.last_read[i]=np.nan
                continue
            new=seq[i]!=self.last_seq[i]
            if new:
                read_dt=0. if np.isnan(self.last_read[i]) else read[i]-self.last_read[i]
                if read_dt<0 or read_dt>.08:
                    self.bias[i]=0.;read_dt=0.
                if enabled[i] and read_dt>0:
                    # The pinned neck/head pitch axes oppose each other.
                    # At an actual native neck-command limit, a remaining
                    # outward neck error cannot be corrected by that command.
                    # Assign only its optical pitch component to head_pitch.
                    # This is command feedback, never a joint/state override.
                    neck_blocked=bool(
                        self.sent[i,0]<=-BOUND[0]+1e-9 and error[i,0]<-.005
                        or self.sent[i,0]>=BOUND[0]-1e-9 and error[i,0]>.005)
                    if neck_blocked:
                        integral_error[i,1]=error[i,1]-error[i,0]
                        redistributed[i]=True
                    e=np.where(abs(integral_error[i])>.005,integral_error[i],0.)
                    prior=self.bias[i].copy()
                    # Clamp the integral to both its correction budget and the
                    # remaining source command box: no windup behind a limit.
                    self.bias[i]=np.clip(self.bias[i]+.35*e*read_dt,
                        np.maximum(-.35,-BOUND-nominal[i]),np.minimum(.35,BOUND-nominal[i]))
                    delta[i]=self.bias[i]-prior;used[i]=True
                self.last_seq[i]=seq[i];self.last_read[i]=read[i]
            if not enabled[i]:self.bias[i]=0.
            wanted=np.clip(nominal[i]+self.bias[i],-BOUND,BOUND)
            self.sent[i]+=np.clip(wanted-self.sent[i],-slew_rad_s*dt,slew_rad_s*dt)
        return self.sent.copy(),{'desired_absolute_q':nominal+HOME,'error_rad':error,
            'integral_error_rad':integral_error,'pitch_error_redistributed':redistributed,'bias_rad':self.bias.copy(),
            'integrated_new_read':used,'bias_delta_rad':delta,'same_read_s':read.copy(),'same_read_seq':seq.copy(),
            'slew_rad_s':slew_rad_s,'valid':valid.copy(),'actual_native_command':self.sent.copy()}
