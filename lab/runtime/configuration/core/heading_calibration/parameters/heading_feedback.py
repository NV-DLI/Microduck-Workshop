"""Heading feedback: Microduck simulation and control."""
import math
import numpy as np

K_HEADING,K_RATE,TAU_HEADING,TAU_RATE,WZ_MAX=2.5,.25,.12,.10,.5
MAX_READ_DT,MAX_AGE_S,MAX_POLICY_DT,MIN_GRAVITY_NORM=.025,.08,.025,.5
FAULTS={0:'none',1:'invalid_or_unready_packet',2:'nonfinite_or_null_gravity_or_gyro',3:'stale_future_or_noncausal_clock',
        4:'sequence_or_read_not_monotone',5:'changed_duplicate_read',6:'read_gap_exceeds_0p025',7:'policy_clock_gap'}


def wrap(a):return math.atan2(math.sin(a),math.cos(a))


class BackwardHeadingFeedback:
    def __init__(self,*,vx,start_step,move_steps,desired_heading):
        self.vx=np.asarray(vx,float);self.start=np.asarray(start_step,np.int64);self.move=np.asarray(move_steps,np.int64);n=len(self.vx)
        self.desired=np.asarray(desired_heading,float)
        if self.start.shape!=(n,)or self.move.shape!=(n,)or self.desired.shape!=(n,)or not np.isfinite(self.vx).all()or not np.isfinite(self.desired).all()\
                or (abs(self.desired)>math.pi/2).any()or (self.move<=0).any():raise ValueError('explicit per-world schedule and desired heading required')
        self.started=np.zeros(n,bool);self.h=np.zeros(n);self.h_lp=np.zeros(n);self.rate_lp=np.zeros(n)
        self.last_seq=np.full(n,-1,np.int64);self.last_read=np.full(n,np.nan);self.last_payload=[None]*n
        self.fault=np.zeros(n,np.int64);self.fault_step=np.full(n,-1,np.int64);self.last_now=None

    def _fail(self,i,code,step):
        if self.fault[i]==0:self.fault[i]=code;self.fault_step[i]=step

    def update(self,*,step,now_s,gyro_body,gravity_body,seq,sample_s,read_s,encoder_read_s,valid):
        n=len(self.vx);now=float(now_s)
        gyro=np.asarray(gyro_body,float);grav=np.asarray(gravity_body,float);seq=np.asarray(seq);sample=np.asarray(sample_s,float)
        read=np.asarray(read_s,float);enc=np.asarray(encoder_read_s,float);valid=np.asarray(valid,bool)
        if gyro.shape!=(n,3)or grav.shape!=(n,3)or any(x.shape!=(n,)for x in (seq,sample,read,enc,valid)):raise ValueError('explicit per-world public input shape required')
        dt=None if self.last_now is None else now-self.last_now
        clock_ok=math.isfinite(now)and (dt is None or 0<dt<=MAX_POLICY_DT+1e-12)
        if math.isfinite(now)and (self.last_now is None or now>self.last_now):self.last_now=now
        vel=np.zeros((n,3));rate=np.full(n,np.nan);integrated=np.zeros(n,bool);read_dt=np.zeros(n);active=np.zeros(n,bool)
        for i in range(n):
            if not(self.start[i]<=step<self.start[i]+self.move[i])or self.fault[i]:continue
            if not clock_ok:self._fail(i,7,step);continue
            if not valid[i]:self._fail(i,1,step);continue
            g=grav[i];w=gyro[i];gn=float(np.linalg.norm(g))if np.isfinite(g).all()else math.nan
            if not(np.isfinite(w).all()and math.isfinite(gn)and gn>=MIN_GRAVITY_NORM):self._fail(i,2,step);continue
            s=seq[i];ints=isinstance(s,(int,np.integer))and not isinstance(s,(bool,np.bool_))
            if not(ints and s>=0 and all(math.isfinite(x)and x>=0 for x in (sample[i],read[i],enc[i]))and enc[i]==read[i]
                   and sample[i]<=read[i]+1e-9 and 0<=now-read[i]<=MAX_AGE_S and 0<=now-sample[i]<=MAX_AGE_S):
                self._fail(i,4 if not ints or s<0 else 3,step);continue
            r=float(np.dot(w,-g/gn));payload=(float(sample[i]),float(read[i]),tuple(w.tolist()),tuple(g.tolist()))
            if not self.started[i]:
                self.started[i]=True;self.h[i]=self.desired[i];self.h_lp[i]=self.desired[i];self.rate_lp[i]=r
            else:
                if s==self.last_seq[i]:
                    if payload!=self.last_payload[i]:self._fail(i,5,step);continue
                elif s<self.last_seq[i]or read[i]<=self.last_read[i]:self._fail(i,4,step);continue
                else:
                    d=read[i]-self.last_read[i]
                    if d>MAX_READ_DT+1e-12:self._fail(i,6,step);continue
                    self.h[i]=wrap(self.h[i]-r*d);integrated[i]=True;read_dt[i]=d
                self.h_lp[i]+=(1.-math.exp(-dt/TAU_HEADING))*(self.h[i]-self.h_lp[i])
                self.rate_lp[i]+=(1.-math.exp(-dt/TAU_RATE))*(r-self.rate_lp[i])
            self.last_seq[i]=int(s);self.last_read[i]=float(read[i]);self.last_payload[i]=payload
            rate[i]=r;active[i]=True
            vel[i]=[self.vx[i],0.,max(-WZ_MAX,min(WZ_MAX,K_HEADING*self.h_lp[i]-K_RATE*self.rate_lp[i]))]
        return vel,{'heading_rad':np.where(self.started,self.h,np.nan),'heading_lp_rad':np.where(self.started,self.h_lp,np.nan),
            'rate_lp_rad_s':np.where(self.started,self.rate_lp,np.nan),'yaw_rate_rad_s':rate,'integrated':integrated,'read_dt_s':read_dt,
            'active':active,'fault_code':self.fault.copy(),'fault_step':self.fault_step.copy(),'velocity_cmd':vel.copy(),
            'policy_dt_s':np.array(np.nan if dt is None else dt),'now_s':np.array(now)}
