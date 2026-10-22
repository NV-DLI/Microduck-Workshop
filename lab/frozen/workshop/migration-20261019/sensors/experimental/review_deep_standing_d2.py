"""Independent CPU review of teacher history, head authority and target cost."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch

M=Path(__file__).resolve().parents[2];D=M/'gait/experiments/deep_standing_d2';O=M.parents[1]/'outputs/migration-20261019'
sys.path.insert(0,str(M/'gait'))
from bounded_actions import BoundedActions,load_limits
spec=importlib.util.spec_from_file_location('_d2_contract_review',D/'contract.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
torch.manual_seed(975191);n=1024
limits=torch.tensor(load_limits()['physical_output_limits_rad']);sl=limits[None].expand(n,-1,-1).clone()
home=limits.mean(-1)[None].expand(n,-1).clone();bias=torch.randn(n,14)*.02;passive=torch.randn(n,14)*.003
q=home-passive;previous=home.clone();teacher=torch.randn(n,14)*1.5;head=torch.randn(n,4)*.3
lower,upper=mod.box(limits,.035,sl,passive);yes=torch.ones(n,dtype=torch.bool);zero=torch.zeros(n,14)
request,history,detail=mod.compose(teacher,zero,head,home=home,bias=bias,lower=lower,upper=upper,ready=yes,task_active=yes)
combined=teacher.clone();combined[:,mod.HEAD]=head+.15*torch.tanh(teacher[:,mod.HEAD])
def execute(raw):
    limiter=BoundedActions(limits,margin_rad=.035,max_target_rate_rad_s=8.);limiter.previous=previous.clone()
    return limiter.apply(raw,scale=1.,offset=home,encoder_bias=bias,q_servo=q,q_passive=passive,servo_limits=sl)[1]
a=execute(combined);b=execute(request)
checks={'zero_request_with_bias_exact':torch.equal(request,combined),
    'zero_history_teacher_exact':torch.equal(history,teacher),
    'zero_final_target_exact':torch.equal(a['executed_output_target_rad'],b['executed_output_target_rad'])}
target_roundoff=float(abs(a['requested_output_target_rad']-detail['unprojected_target']).max())
old_cost=(a['requested_output_target_rad']-a['executed_output_target_rad']).square().sum(-1)
new_cost=(detail['unprojected_target']-b['executed_output_target_rad']).square().sum(-1)
cost_roundoff=float(abs(old_cost-new_cost).max())
checks['target_cost_mathematical_semantics_with_float32_roundoff']=target_roundoff<=1e-6 and cost_roundoff<=1e-4
noise=torch.randn(n,14)*2
req,h,info=mod.compose(teacher,noise,head,home=home,bias=bias,lower=lower,upper=upper,ready=yes,task_active=yes)
checks.update(head_execution_request_original=torch.equal(req[:,mod.HEAD],combined[:,mod.HEAD]),
    head_raw_history_original=torch.equal(h[:,mod.HEAD],teacher[:,mod.HEAD]),
    leg_history_exact=torch.equal(h,teacher+info['delta']),
    only_ten_leg_delta=not bool(info['delta'][:,mod.HEAD].any()),
    physical_delta_bounded=bool((info['delta'].abs()<=.15).all()))
mixed=yes.clone();mixed[::2]=False
_,_,g=mod.compose(teacher,noise,head,home=home,bias=bias,lower=lower,upper=upper,ready=yes,task_active=mixed)
checks['preparation_delta_zero']=not bool(g['delta'][~mixed].any())
req,h,_=mod.compose(teacher,noise,head,home=home,bias=bias,lower=lower,upper=upper,ready=mixed,task_active=yes)
checks['unready_request_history_zero']=not bool(req[~mixed].any()or h[~mixed].any())
assert all(checks.values()),checks
report={'schema':'microduck.deep-standing-d2-independent-cpu-review.v1','passed':True,'checks':checks,
    'synthetic_encoder_bias_rows':n,'same_real_state_evidence':str(O/'deep-standing-d2-cpu-contract-v1.json'),
    'original_cost_arithmetic_bitwise_equal':target_roundoff==0.,
    'cost_roundoff':{'unprojected_target_max_abs_rad':target_roundoff,'squared_sum_cost_max_abs':cost_roundoff,
        'cause':'Original computes(combined+home)-bias; new metadata computes combined+(home-bias). Request/actual target remain bitwise equal atzero delta; restored cost is mathematically equal with float32 reassociation, not claimed bitwise.'},
    'evidence':{str(p):sha(p)for p in (Path(__file__),D/'contract.py',D/'environment.py',D/'train.py',D/'plan.json',O/'deep-standing-d2-cpu-contract-v1.json')},
    'source_review':{'head_mapping_once':'train explicitly selects precision_recipe_v3.BoundedTrainingAction; compose implements the old head mapper exactly once.',
        'history':'After final limiter, ActionManager.action and BODY50.previous_raw are restored to original teacher raw plus actual bounded leg delta before another observation.',
        'target_excess':'Restored to original unprojected combinedteacher+delta versus actualtarget, so teacher projection does not silently eliminate cost.',
        'state_initialization':'Fresh residual Adam, critic and input normalizer inherited; initial.pt equality checks are present. Actual technical audit is separate.',
        'unused_head_gaussian':'Four sampled residual head coordinates affect no physical target/history. They remain in PPO log-probability and can add gradient variance; this is not executed head noise.'},
    'physical_success_claimed':False}
out=O/'deep-standing-d2-independent-cpu-review-v1.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(checks))
