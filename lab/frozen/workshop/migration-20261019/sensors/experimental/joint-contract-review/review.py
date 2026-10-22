"""CPU/source audit of the official fourteen policy joint channels."""
import ast,hashlib,json,math,re,sys
from pathlib import Path
M=Path('@COURSE@/workshop/migration-20261019');S=M/'sensors';L=M/'runtime/IsaacLab-5088b326c7fc81caa7899395b0176ad08b89e9e1';D=Path(__file__).parent
asset=L/'source/isaaclab_assets/isaaclab_assets/robots/microduck.py';flat=L/'source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/flat_env_cfg.py';obs=L/'source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/mdp/observations.py';act=obs.with_name('actions.py');joint=L/'source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py';bam=L/'source/isaaclab_newton/isaaclab_newton/assets/articulation/actuator_control.py'
model=S/'experimental/voltage-source-review/duck-control__src__model.rs';rustobs=S/'sources/body-imu/duck-control__src__obs.rs';bus=S/'sources/body-imu/duck-control__src__bus.rs';ctrl=S/'sources/body-imu/robotd__src__control.rs';proto=D/'duck-ipc-proto__src__lib.rs'
def assign(path,name):
 for n in ast.parse(path.read_text()).body:
  if isinstance(n,ast.AnnAssign)and isinstance(n.target,ast.Name)and n.target.id==name:return ast.literal_eval(n.value)
  if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==name for t in n.targets):return ast.literal_eval(n.value)
 raise KeyError(name)
def rustarray(path,name):
 text=path.read_text();s=re.search(r'pub const '+name+r'[^=]*=\s*\[(.*?)\];',text,re.S)[1];s=re.sub(r'//[^\n]*','',s)
 return ast.literal_eval('['+s+']')
policy=assign(asset,'MICRODUCK_JOINT_NAMES');names=rustarray(proto,'JOINT_NAMES');ids=rustarray(proto,'JOINT_IDS');home15=rustarray(model,'DEFAULT_POSITION');home14=assign(M/'skills/contract.py','HOME_POSE');select=[i for i in range(15)if i!=9]
checks={
 'official_runtime_without_mouth_equals_official_isaac_order':[names[i]for i in select]==policy,
 'course_skill_order_equals_official':list(assign(M/'skills/contract.py','POLICY_JOINTS'))==policy,
 'runtime_home_without_mouth_equals_course_home':[home15[i]for i in select]==list(home14),
 'both_hip_yaw_default_zero':home14[0]==home14[9]==0,
 'explicit_native_action_order':'joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True, scale=1.0, use_default_offset=True'in flat.read_text(),
 'all_native_observation_servo_order':'_SERVO_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_JOINT_NAMES, preserve_order=True)'in flat.read_text(),
 'native_encoder_adds_passive_and_bias_then_subtracts_home':'joint_pos = joint_pos + env.action_manager.get_term(action_name).encoder_bias'in obs.read_text()and'joint_pos = joint_pos + env.scene[backlash_cfg.name].data.joint_pos.torch[:, backlash_cfg.joint_ids]'in obs.read_text(),
 'native_action_subtracts_bias_once':'target = self.processed_actions - self.encoder_bias'in act.read_text(),
 'base_position_action_affine_scale_home':'self._processed_actions = self._raw_actions * self._scale'in joint.read_text()and'self._processed_actions.add_(self._offset)'in joint.read_text(),
 'hardware_same_positive_affine_position_no_side_sign':'sensors.positions[joint] = (2.0 * PI * position as f64 / 4096.0) - PI;'in bus.read_text(),
 'hardware_actions_home_plus_scale':'targets[joint] = DEFAULT_POSITION[joint] + scale * offsets[joint];'in ctrl.read_text(),
 'hardware_previous_action_is_raw':'*previous_action = *last_action;'in rustobs.read_text(),
 'only_knees_homing_offset':'pub const KNEE_IDS: [u8; 2] = [13, 23];'in model.read_text(),
 'native_backlash_matches_each_named_servo_sibling':'twin_label = f"{parent}/{template.format(name)}"'in bam.read_text(),
}
for slot in range(14):
 a=[0.]*14;a[slot]=.01;b=[0.]*15
 for j,x in enumerate(a):b[select[j]]=x
 assert [b[i]for i in select]==a and b[9]==0
checks['all14_unique_impulse_scatter_gather_roundtrip']=True
assert all(checks.values()),checks
rows=[{'policy_index':j,'hardware_array_index':i,'joint':names[i],'dynamixel_id':ids[i],'home_rad':home15[i],'position_obs_index':6+j,'velocity_obs_index':20+j,'previous_raw_action_obs_index':34+j}for j,i in enumerate(select)]
raw=-1.994;limit=(-math.pi/6,math.radians(25));turns=[raw+2*math.pi*k for k in range(-2,3)]
paths=[Path(__file__),asset,flat,obs,act,joint,bam,model,rustobs,bus,ctrl,proto,M/'skills/contract.py',M/'gait/bounded_actions.py',M/'acceptance/body50_training/physics_producer.py']
report={'schema':'microduck.official-policy-joint-channel-review.v1','checks':checks,'checks_passed':all(checks.values()),'policy_rows':rows,
 'fixed_isaac_revision':'5088b326c7fc81caa7899395b0176ad08b89e9e1','fixed_runtime_revision':'53230463927095ea890f231d29089bb0a538c06f',
 'proto_url':'https://raw.githubusercontent.com/pollen-robotics/microduck/53230463927095ea890f231d29089bb0a538c06f/duck-ipc-proto/src/lib.rs',
 'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths},
 'hip_yaw_diagnostic':{'reported_raw_example_rad':raw,'nominal_home_rad':0.,'official_isaac_scale':1.,'example_joint_limits_rad':limit,'example_wrap_candidates_rad':turns,'any_wrap_candidate_inside_limit':any(limit[0]<=x<=limit[1]for x in turns),'negating_raw_inside_limit':limit[0]<=-raw<=limit[1],
 'interpretation':'Neither one mouth-index shift nor a left/right sign convention is present in the reviewed official policy mapping. A raw position target is not a measured joint angle: the official unbounded policy action can request beyond a joint stop to drive PD/BAM torque. Such requests can harm a separately bounded controller; magnitude alone does not prove coordinate corruption.'},
 'scope':'Source and CPU channel contract only; no new GPU run, no axis/local-frame/USD calibration claim. Runtime source explicitly describes Alpha15 servos while the policy omits mouth; current Beta sensor profile does not prove a new real-body kinematic calibration.',
 'followup':'Compare actual solver joint names/axes/limits and signal perturbation before changing any yaw limit or action sign.'}
out=M.parents[1]/'outputs/migration-20261019/official-policy-joint-channel-review-v1.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':sum(checks.values()),'total':len(checks),'hip_yaw_rows':[rows[i]for i in (0,9)],'output':str(out)}))
