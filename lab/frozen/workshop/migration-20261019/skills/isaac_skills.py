#!/usr/bin/env python3
"""Train, validate and interact with MicroDuck skills in Isaac Sim / Newton."""
from __future__ import annotations
import argparse, hashlib, json, time, sys, os
from datetime import datetime
from pathlib import Path
from isaaclab.app import launch_simulation
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('mode',choices=['train','eval','play'])
p.add_argument('skill',choices=['basketball','jump','dance','backflip','arabesque'])
p.add_argument('--source',choices=['community','trained','research','zero'],default='trained')
p.add_argument('--checkpoint',type=Path)
p.add_argument('--export-dir',type=Path)
p.add_argument('--stochastic',action='store_true',help='Sample the trained PPO distribution instead of its deterministic mean.')
p.add_argument('--awake',action='store_true',help='Request awake imported bodies for diagnostic comparison; backend support varies.')
p.add_argument('--ball-material',choices=['simple','detailed'],default='simple')
p.add_argument('--view-profile',choices=['balanced','quality','fast'],default='balanced')
p.add_argument('--view-fps',type=int,choices=[30,50,60],default=60)
p.add_argument('--legacy-view',action='store_true',help='Reproduce the previous 720p/50 Hz viewport for comparison.')
p.add_argument('--profile-python',type=Path,help='Write a cProfile of the active inference loop for local performance diagnosis.')
p.add_argument('--verify-pause',action='store_true',help='Check direct physics poses and joints stay frozen while the GUI is paused.')
p.add_argument('--training-video',type=Path,help='Record actual parallel PPO rollouts, with a grid overview camera.')
p.add_argument('--training-video-start-iteration',type=int,default=300)
p.add_argument('--training-video-seconds',type=float,default=12.0)
p.add_argument('--video',type=Path,help='Encode physical frames at 50 fps; one environment only.')
p.add_argument('--num-envs',type=int,default=None)
p.add_argument('--episode-seconds',type=float,default=None,help='Training horizon override, retained with the run configuration.')
p.add_argument('--finite-horizon',action='store_true',help='Treat the training time limit as a terminal episodic objective, without value bootstrap.')
p.add_argument('--training-gravity-scale',type=float,default=None,help='Training curriculum only. Validation and play always use normal gravity.')
p.add_argument('--iterations',type=int,default=2000)
p.add_argument('--initial-std',type=float,default=.3,help='Initial Gaussian action exploration for a fresh PPO actor.')
p.add_argument('--entropy-coef',type=float,default=.003)
p.add_argument('--freeze-action-std',action='store_true',help='Keep the resumed Gaussian policy standard deviations fixed while optimizing the actor mean.')
p.add_argument('--save-interval',type=int,default=250)
p.add_argument('--adapt-arabesque-observations',action='store_true',help='Warm-start a legacy native lift actor, enabling new static-balance observations with zero initial input weights.')
p.add_argument('--adapt-backflip-observations',action='store_true',help='Enable native flight-height and landing-state observations with zero initial input weights.')
p.add_argument('--jump-reset',choices=['mixed','standing'],default=None)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--success-threshold',type=float,default=.9)
p.add_argument('--duration',type=float,default=30)
p.add_argument('--push-speed',type=float,default=0.0,help='Evaluation only: reset ball XY velocity to a random direction at this speed every 2 s.')
p.add_argument('--output',type=Path)
p.add_argument('--run-name',default='')
p.add_argument('--completion-file',type=Path)
p.add_argument('--actuator',choices=['bam'],default='bam',help='Official Newton-native DriveBam')
p.add_argument('--nominal',action='store_true',help='Diagnostic only: disable physical randomization; ordinary train/eval retain it')
p.add_argument('--robot-variant',choices=['servo14','backlash28'],help='New training defaults backlash28; community/historical checkpoints default servo14; saved lineage takes precedence')
p.add_argument('--robot-usd',type=str,help='Explicit local copy matching --robot-variant and collision variant')
p.add_argument('--out-root',type=Path,default=Path(__file__).resolve().parent/'runs')
p.add_argument('--reference-root',type=Path,default=Path(os.environ.get('MICRODUCK_REFERENCE_ROOT','@LEGACY@/microduck-ros2-isaac')),help='Read-only community weights and visual assets; does not import the historical runtime')
p.add_argument('--basketball-recipe',choices=['legacy','robust','reset_curriculum'])
p.add_argument('--jump-objective',choices=['legacy','discovery','physical'],default=None)
p.add_argument('--arabesque-objective',choices=['legacy','static','hold'],default=None)
p.add_argument('--backflip-objective',choices=['legacy','launch_spin','compact_flight','compact_landing','landing_upright'],default=None)
p.add_argument('--backflip-reset',choices=['mixed','standing'],default=None)
p.add_argument('--backflip-recovery-curriculum',action='store_true',help='Train recovery after >=320 degree feet contacts; eval/play keep the original complete-turn acceptance gate.')
p.add_argument('--backflip-strict-finish',action='store_true',help='After learning recovery, emphasize upright takeoff and sagittal rotation; reduce partial-turn recovery credit.')
p.add_argument('--prime-reset-delay',action=argparse.BooleanOptionalAction,default=None,help='Initialize BAM command history consistently at cold and automatic resets. Default on for backflip.')
p.add_argument('--reset-noise',type=float,default=1.0)
p.add_argument('--root-height',type=float)
p.add_argument('--handoff',action='store_true',help='Use the public jump-to-standing handoff (0.22 s start, 0.14 s smooth blend).')
p.add_argument('--preserve-latest',action='store_true',help='Save this run without replacing the default research training pointer.')
p.add_argument('--resume',type=Path,help='Explicit continuation of an Isaac scratch run; default always random initialization.')
# Upstream add_launcher_args imports KitLauncher even for a Newton-only CLI.
# With the full Isaac Sim wheel installed on aarch64, that import can exit at
# the libgomp preload check before argument parsing. Declare our small public
# launcher surface locally; launch_simulation still validates the runtime.
p.add_argument('--device',default='cuda:0')
p.add_argument('--viz',dest='visualizer',nargs='*',choices=['none','kit','newton_gl','newton_rtx','rerun','viser'],default=[])
p.add_argument('--livestream',type=int,choices=[0,1,2],default=0)
p.add_argument('--kit_args',default='')
p.add_argument('--max_visible_envs',type=int)
a=p.parse_args()
for checkpoint in (a.checkpoint,a.resume):
    if checkpoint is not None and not checkpoint.is_file():p.error(f'Checkpoint not found: {checkpoint}')
if a.initial_std<=0:p.error('--initial-std must be positive')
if not .9<=a.success_threshold<=1:p.error('Skill acceptance cannot be weakened: --success-threshold must be >=0.9')
if a.training_gravity_scale is not None and (a.mode!='train' or not 0<a.training_gravity_scale<=1):p.error('Gravity curriculum is train-only, with scale in (0,1]')
if a.video or a.training_video:a.enable_cameras=True
if a.training_video and a.mode!='train':p.error("--training-video requires train")
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from skill_env import SkillEnv,SkillEnvCfg,T
from ppo_cfg import MicroDuckVelocityPPORunnerCfg
from isaaclab.utils.io import dump_yaml
SOURCE_DIR=Path(__file__).resolve().parent
ROOT=a.reference_root.resolve()
OUT=a.out_root.resolve()
torch.set_num_threads(1)
# PPO may use TF32 for throughput; acceptance/play must match the CPU FP32
# TorchScript/ONNX exports. Export validation also checks strict FP32 on CUDA.
torch.set_float32_matmul_precision('high' if a.mode=='train' else 'highest')
torch.backends.cuda.matmul.allow_tf32=a.mode=='train'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def cfg_agent():
    c=MicroDuckVelocityPPORunnerCfg()
    c.seed=a.seed;c.device=a.device;c.clip_actions=None;c.num_steps_per_env=24
    c.actor.hidden_dims=[256,128,64];c.critic.hidden_dims=[256,128,64]
    c.actor.distribution_cfg.init_std=a.initial_std
    c.algorithm.entropy_coef=a.entropy_coef;c.algorithm.num_learning_epochs=4
    c.save_interval=a.save_interval
    return c  # RSL-RL 5 uses the current actor/critic configuration directly.

class OnnxActor:
    def __init__(self,path,n):
        import onnxruntime as ort
        opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(path),opts,providers=['CPUExecutionProvider'])
        self.n=n;self.inputs=self.session.get_inputs();self.h=np.zeros((1,n,256),np.float32);self.c=self.h.copy()
        self.batched = self.inputs[0].shape[0] != 1
    def reset(self,ids=None):
        if ids is None:self.h.fill(0);self.c.fill(0)
        else:self.h[:,ids,:]=0;self.c[:,ids,:]=0
    def __call__(self,obs):
        x=obs.detach().cpu().numpy().astype(np.float32)
        def run(x,h,c):
            feed={self.inputs[0].name:x}
            if len(self.inputs)==3:feed.update({self.inputs[1].name:h,self.inputs[2].name:c})
            return self.session.run(None,feed)
        if self.n==1 or self.batched:
            out=run(x,self.h,self.c)
            if len(out)==3:self.h,self.c=out[1:]
            return torch.from_numpy(out[0]).to(obs.device)
        actions=[]
        for i in range(self.n):
            out=run(x[i:i+1],self.h[:,i:i+1],self.c[:,i:i+1]);actions.append(out[0])
            if len(out)==3:self.h[:,i:i+1]=out[1];self.c[:,i:i+1]=out[2]
        return torch.from_numpy(np.concatenate(actions)).to(obs.device)


class CommunityBackflipActor:
    """The published assisted v88 actor, tested here without assistance.

    Load tensors only. Its 61D normalizer and ELU MLP match upstream rsl_rl;
    the cube, spotter and recovery handoff are not supplied in this flat test.
    """
    def __init__(self,path,device):
        if sha(path)!='86cc4ec03839d2d1407be63c9b17a22bd96c5368bd0c2d05fc06b08014c8d598':
            raise ValueError('Community backflip checkpoint differs from upstream release hash')
        state=torch.load(path,map_location=device,weights_only=True)['actor_state_dict']
        self.mean=state['obs_normalizer._mean'];self.std=state['obs_normalizer._std']
        self.layers=[]
        for i in (0,2,4,6):self.layers.append((state[f'mlp.{i}.weight'],state[f'mlp.{i}.bias']))
    def __call__(self,obs):
        x=(obs-self.mean)/(self.std+.01)
        for i,(weight,bias) in enumerate(self.layers):
            x=torch.nn.functional.linear(x,weight,bias)
            if i<3:x=torch.nn.functional.elu(x)
        return x


def main():
    env=None;video_process=None;video_frames=0;training_video=None;runtime_context=None;app=None;accounting=None
    startup_started=time.monotonic()
    try:
        if a.mode!='train' and a.source in ('trained','research') and a.checkpoint is None:
            selection_path=OUT/a.skill/('latest_training.json' if a.source=='research' else 'latest.json')
            if not selection_path.exists():raise ValueError(f'No validated Newton {a.skill} model. Train and validate it, or explicitly use --source research to inspect the latest unvalidated training run.')
            selected=json.loads(selection_path.read_text())
            a.awake=a.awake or selected.get('disable_sleep',False)
            a.stochastic=a.stochastic or selected.get('stochastic_evaluation_default',False)
        runtime={}
        runtime_checkpoint=a.checkpoint or a.resume
        if runtime_checkpoint and (runtime_checkpoint.parent/'runtime.json').exists():
            runtime=json.loads((runtime_checkpoint.parent/'runtime.json').read_text())
        elif a.mode!='train' and a.source in ('trained','research') and a.checkpoint is None:
            runtime=selected
        from backlash_contract import resolve_variant
        a.robot_variant,variant_reason=resolve_variant(a.robot_variant,runtime,mode=a.mode,source=a.source,
            has_checkpoint=bool(runtime_checkpoint or (a.mode!='train' and a.source in ('trained','research'))))
        a.jump_objective=a.jump_objective or runtime.get('jump_objective') or 'physical'
        a.jump_reset=a.jump_reset or runtime.get('jump_reset') or 'mixed'
        a.arabesque_objective=a.arabesque_objective or runtime.get('arabesque_objective') or ('hold' if a.mode=='train' and not a.resume else 'legacy')
        a.backflip_objective=a.backflip_objective or runtime.get('backflip_objective') or 'legacy'
        a.backflip_reset=a.backflip_reset or runtime.get('backflip_reset') or 'mixed'
        a.freeze_action_std=a.freeze_action_std or runtime.get('freeze_action_std',False)
        recipe=a.basketball_recipe or runtime.get('basketball_recipe') or ('robust' if a.mode=='train' else 'legacy')
        cfg=SkillEnvCfg();cfg.basketball_recipe=recipe;cfg.skill=a.skill;cfg.actuator=a.actuator;cfg.seed=a.seed
        cfg.domain_randomization=not a.nominal;cfg.robot_usd=a.robot_usd;cfg.robot_variant=a.robot_variant
        cfg.scene.num_envs=a.num_envs or (512 if a.mode=='train' else 1)
        cfg.root_height=a.root_height or (.115 if a.source=='community' and a.skill=='jump' else .125)
        cfg.sim.device=a.device;cfg.reset_noise=a.reset_noise;cfg.jump_objective=a.jump_objective;cfg.jump_reset=a.jump_reset;cfg.disable_sleep=a.awake
        cfg.arabesque_objective=a.arabesque_objective
        cfg.backflip_objective=a.backflip_objective
        cfg.backflip_reset=a.backflip_reset
        cfg.backflip_recovery_curriculum=a.mode=='train' and (a.backflip_recovery_curriculum or runtime.get('backflip_recovery_curriculum',False))
        cfg.backflip_strict_finish=a.backflip_strict_finish or runtime.get('backflip_strict_finish',False)
        cfg.prime_reset_delay=a.prime_reset_delay if a.prime_reset_delay is not None else runtime.get('prime_reset_delay',a.skill=='backflip')
        gravity_scale=(a.training_gravity_scale if a.training_gravity_scale is not None else runtime.get('training_gravity_scale',1.0)) if a.mode=='train' else 1.0
        cfg.sim.gravity=(0.,0.,-9.81*gravity_scale)
        cfg.is_finite_horizon=a.finite_horizon or runtime.get('finite_horizon',False)
        cfg.episode_length_s=4.0 if a.skill=='backflip' else 3.5 if a.skill=='jump' else (8 if a.mode=='train' else max(a.duration+1,3600 if a.mode=='play' else 1))
        if a.mode=='train' and a.skill=='basketball' and recipe in ('robust','reset_curriculum'):cfg.episode_length_s=20.0
        if a.mode=='train' and a.skill=='arabesque' and a.arabesque_objective in ('static','hold'):cfg.episode_length_s=20.0
        if a.mode=='train' and a.episode_seconds:cfg.episode_length_s=a.episode_seconds
        cfg.evaluation=a.mode!='train'
        cfg.record_video=a.video is not None or a.training_video is not None
        if a.training_video:cfg.scene.env_spacing=.6
        if a.video and cfg.scene.num_envs!=1:raise ValueError('Video requires --num-envs 1')
        if a.mode=='play' or a.video or a.training_video:
            from isaaclab.sim import RenderCfg
            from isaac_render_profile import render_config
            cfg.sim.render=RenderCfg(rendering_mode='performance') if a.legacy_view else render_config(a.view_profile)
            if a.video or a.training_video:
                cfg.sim.render.rendering_mode='balanced';cfg.sim.render.enable_dl_denoiser=True
            cfg.viewer.eye=(.65,-.65,.55) if a.skill=='basketball' else (.45,-.45,.3);cfg.viewer.lookat=(0,0,.30 if a.skill=='basketball' else .12)
        # Modern Lab can train Newton without starting Kit. Dynamic scene
        # cameras and the native GUI explicitly require Kit only when used.
        a.require_kit=bool(a.mode=='play' or a.video or a.training_video)
        if a.mode=='play' and not a.visualizer:a.visualizer=['kit']
        runtime_context=launch_simulation(cfg,a)
        runtime_context.__enter__()
        if a.require_kit:
            import omni.kit.app
            app=omni.kit.app.get_app()
        env=SkillEnv(cfg)
        env.reset()
        from skill_randomization import observed_randomization
        env.dr_evidence["actual_initial_parameters"]=observed_randomization(env)
        env.dr_evidence['variant_selection_reason']=variant_reason
        from newton_backend import backend_evidence
        physics_info=backend_evidence(env.sim)
        print('ISAAC_NEWTON_BACKEND='+json.dumps(physics_info),flush=True)
        prop_info=None
        if env.ball is not None and (a.mode=='play' or a.video):
            from isaac_props import attach_basketball
            import omni.usd
            stage=omni.usd.get_context().get_stage()
            prop_info=attach_basketball(stage, ROOT/('assets/props/basketball/basketball_detailed.usdc' if a.ball_material=='detailed' else 'assets/props/basketball/basketball.usdc'))
            print('ISAAC_PROP_INFO='+json.dumps(prop_info),flush=True)
        if env.ball is not None and a.training_video:
            from isaac_props import attach_basketball
            import omni.usd
            stage=omni.usd.get_context().get_stage()
            origins=env.scene.env_origins.detach().cpu().numpy()
            visible=np.flatnonzero(np.linalg.norm(origins[:,:2],axis=1)<5.)
            for idx in visible:
                path=f'/World/envs/env_{idx}/Ball'
                attach_basketball(stage,ROOT/'assets/props/basketball/basketball.usdc',path)
                stage.GetPrimAtPath(path+'/CommunityVisual').SetInstanceable(True)
            print(f'ISAAC_TRAINING_PROPS={len(visible)} smooth shared basketball visuals; all analytic collision spheres unchanged',flush=True)
        print('ISAAC_SKILLS_READY',a.mode,a.skill,flush=True)
        c=cfg_agent()
        if a.mode=='train':
            run=OUT/a.skill/(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+('_'+a.run_name if a.run_name else ''))
            run.mkdir(parents=True,exist_ok=False)
            import shutil
            snapshot=run/'source_snapshot';snapshot.mkdir()
            for file in SOURCE_DIR.glob('*.py'):shutil.copy2(file,snapshot/file.name)
            shutil.copy2(__file__,run/'isaac_skills.py')
            shutil.copy2(SOURCE_DIR/'training_accounting.py',run/'training_accounting.py')
            shutil.copy2(SOURCE_DIR/'newton_backend.py',run/'newton_backend.py')
            (run/"runtime.json").write_text(json.dumps({"basketball_recipe":recipe,"disable_sleep":a.awake,"jump_objective":a.jump_objective,"jump_reset":a.jump_reset,"arabesque_objective":a.arabesque_objective,"backflip_objective":a.backflip_objective,"backflip_reset":a.backflip_reset,"backflip_recovery_curriculum":cfg.backflip_recovery_curriculum,"backflip_strict_finish":cfg.backflip_strict_finish,"prime_reset_delay":cfg.prime_reset_delay,"training_gravity_scale":gravity_scale,"finite_horizon":cfg.is_finite_horizon,"freeze_action_std":a.freeze_action_std,"initial_std":a.initial_std,"training_episode_seconds":cfg.episode_length_s,"physics":physics_info,"domain_randomization":env.dr_evidence,"migration":"native-bam-20261019","robot_variant":a.robot_variant,"robot_variant_selection":variant_reason},indent=2))
            dump_yaml(str(run/'env.yaml'),cfg);dump_yaml(str(run/'agent.yaml'),c)
            wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
            runner=OnPolicyRunner(wrapped,c.to_dict(),str(run),device=a.device)
            runner.logger.writer=None
            if a.resume:
                runner.load(str(a.resume),map_location=a.device)
                from training_accounting import parent_work
                # Measured completed count also handles model_initial.pt,
                # whose serialized iter=0 does not mean an update happened.
                runner.current_learning_iteration=parent_work(a.resume)['iterations']
            if a.freeze_action_std:
                runner.alg.actor.distribution.std_param.requires_grad_(False)
            if a.adapt_arabesque_observations or a.adapt_backflip_observations:
                valid_adaptation=(a.adapt_arabesque_observations and a.skill=='arabesque' and a.arabesque_objective in ('static','hold')) or (a.adapt_backflip_observations and a.skill=='backflip' and a.backflip_objective in ('compact_flight','compact_landing','landing_upright'))
                if not a.resume or not valid_adaptation:
                    raise ValueError('Observation adaptation requires the corresponding native skill resume and objective')
                with torch.no_grad():
                    for model in (runner.alg.actor,runner.alg.critic):
                        model.mlp[0].weight[:,63:72]=0
                        model.obs_normalizer._mean[:,63:72]=0
                        model.obs_normalizer._var[:,63:72]=1
                        model.obs_normalizer._std[:,63:72]=1
                (run/'observation_adaptation.json').write_text(json.dumps(dict(parent=str(a.resume),new_columns=[63,72],action='Zero newly enabled input weights; initialize their normalizer mean=0, variance=1. Other learned parameters and optimizer state retained.'),indent=2)+'\n')
            initial=run/'model_initial.pt';runner.save(str(initial))
            initial_sha=sha(initial)
            start=time.monotonic()
            from training_accounting import TrainingAccounting
            import ast
            objective_ast=ast.parse((SOURCE_DIR/'skill_env.py').read_text())
            reward_hash=hashlib.sha256('\n'.join(ast.dump(node) for node in ast.walk(objective_ast)
                if isinstance(node,ast.FunctionDef) and node.name in ('_get_rewards','_get_dones','_measure')).encode()).hexdigest()
            metadata={'skill':a.skill,'robot_variant':a.robot_variant,'seed':a.seed,'iterations_planned':a.iterations,
                'reward_version':a.skill+'-'+reward_hash[:16],'reward_functions_sha256':reward_hash,
                'source_sha256':{file.name:sha(file) for file in SOURCE_DIR.glob('*.py')},
                'env_yaml_sha256':sha(run/'env.yaml'),'agent_yaml_sha256':sha(run/'agent.yaml'),
                'domain_randomization':env.dr_evidence,'objective':{'basketball_recipe':recipe,
                    'jump':a.jump_objective,'arabesque':a.arabesque_objective,'backflip':a.backflip_objective},
                'shared_gpu':True,'task_acceptance':'not_evaluated','startup_wall_seconds':time.monotonic()-startup_started,
                'rendering_enabled':cfg.record_video,'training_seed':a.seed,'development_seeds':[972001,972002],
                'heldout_seeds':[973001,973002,973003],
                'reward_config_sha256':hashlib.sha256(json.dumps({'functions':reward_hash,'skill':a.skill,'recipe':recipe,
                    'jump':a.jump_objective,'arabesque':a.arabesque_objective,'backflip':a.backflip_objective},sort_keys=True).encode()).hexdigest()}
            hardware_sources={name:sha(SOURCE_DIR.parent/'sensors'/name) for name in (
                'beta_sensor_profile.py','beta_tof_geometry.py','beta_mounts.py','beta_head_imu.py')}
            metadata['hardware_profile_source_sha256']=hardware_sources
            metadata['hardware_profile_sha256']=hashlib.sha256(json.dumps(hardware_sources,sort_keys=True).encode()).hexdigest()
            metadata['hardware_scope']='Beta profile lineage; this privileged 75D skill actor does not consume camera/ToF or prove real-device deployment'
            runtime_source=SOURCE_DIR.parent/'runtime/source-manifest.json'
            runtime_manifest={'actual_runtime':physics_info,'source_manifest_sha256':sha(runtime_source) if runtime_source.exists() else None}
            metadata['runtime_manifest']=runtime_manifest
            metadata['runtime_manifest_sha256']=hashlib.sha256(json.dumps(runtime_manifest,sort_keys=True).encode()).hexdigest()
            accounting=TrainingAccounting(runner,run,a.resume,start,cfg.scene.num_envs,c.num_steps_per_env,
                metadata=metadata,initial_checkpoint=initial,readback=lambda stage:observed_randomization(env,stage))
            print('ISAAC_SKILLS_RUN='+str(run),flush=True)
            if a.training_video:
                from isaac_training_video import TrainingVideo
                if a.training_video_start_iteration*24+round(a.training_video_seconds/.02)>a.iterations*24:
                    raise ValueError('Training video segment extends beyond the requested training run')
                training_video=TrainingVideo(env,a.training_video,a.training_video_start_iteration,a.training_video_seconds,c.num_steps_per_env,backend_evidence(env.sim))
                training_video.info.update(run_directory=str(run),resume_checkpoint=str(a.resume) if a.resume else None,resume_checkpoint_sha256=sha(a.resume) if a.resume else None,initial_optimizer_iteration=int(runner.current_learning_iteration))
            runner.learn(num_learning_iterations=a.iterations,init_at_random_ep_len=False)
            if training_video:training_video.finish()
            final=run/'model_final.pt';runner.save(str(final))
            env.dr_evidence['actual_final_parameters']=observed_randomization(env,'after PPO training')
            from export_evidence import export_and_verify
            export=export_and_verify(runner,a.export_dir or run/'exported',final,env._get_observations()['policy'])
            accounting.finish(export=export)
            summary=dict(skill=a.skill,robot_variant=a.robot_variant,robot_variant_selection=variant_reason,backend='Isaac Lab develop 5088b326 / Newton MJWarp native BAM',physics=backend_evidence(env.sim),domain_randomization=env.dr_evidence,
                initialization='random' if not a.resume else 'resume_isaac_run',pretrained_weights_loaded=False if not a.resume else None,
                resume=str(a.resume) if a.resume else None,resume_checkpoint_sha256=sha(a.resume) if a.resume else None,training_episode_seconds=cfg.episode_length_s,seed=a.seed,num_envs=cfg.scene.num_envs,iterations=a.iterations,
                transitions=cfg.scene.num_envs*a.iterations*c.num_steps_per_env,elapsed_s=time.monotonic()-start,
                initial_checkpoint_sha256=initial_sha,checkpoint=str(final),checkpoint_sha256=sha(final),actuator=a.actuator,
                training_accounting=accounting.summary(),training_ledger=str(run/'training-ledger.json'),reward_version=metadata['reward_version'],export_validation=export,entropy_coef=a.entropy_coef,arabesque_objective=a.arabesque_objective,backflip_objective=a.backflip_objective,backflip_reset=a.backflip_reset,backflip_recovery_curriculum=cfg.backflip_recovery_curriculum,backflip_strict_finish=cfg.backflip_strict_finish,prime_reset_delay=cfg.prime_reset_delay,observation_adaptation=a.adapt_arabesque_observations or a.adapt_backflip_observations,training_episode_schedule={'short_seconds':3.0,'short_fraction':.75,'long_seconds':cfg.episode_length_s} if recipe=='reset_curriculum' else None,training_gravity_scale=gravity_scale,finite_horizon=cfg.is_finite_horizon,freeze_action_std=a.freeze_action_std,
                training_video=str(a.training_video) if a.training_video else None,actor_observation_dim=75,community_observation_dim=61,disable_sleep=a.awake,jump_objective=a.jump_objective,jump_reset=a.jump_reset,initial_std=a.initial_std,basketball_recipe=recipe)
            (run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            summary['validation_status']='not_evaluated'
            if not a.preserve_latest:(OUT/a.skill/'latest_training.json').write_text(json.dumps(summary,indent=2)+'\n')
            print('ISAAC_SKILLS_COMPLETE='+json.dumps(summary),flush=True)
            return
        policy_path=None;actor=None
        if a.source=='community':
            if a.skill=='dance':raise ValueError('This community dance task has no release weights; use arabesque --source community for the published ballet policy.')
            policy_path=ROOT/{'basketball':'community/basketball/artifacts/release/policy.onnx','jump':'community/jump/policy/max_height_jump.onnx','arabesque':'community/arabesque/policy.onnx','backflip':'community/backflip/results/checkpoints/v88-assisted-cube-to-mat-model650/model_650.pt'}[a.skill]
            actor=CommunityBackflipActor(policy_path,a.device) if a.skill=='backflip' else OnnxActor(policy_path,cfg.scene.num_envs)
            if a.handoff:
                if a.skill!='jump':raise ValueError('--handoff is only available for the community jump')
                jump_actor=actor
                stand_actor=OnnxActor(ROOT/'community/jump/handoff/standing_policy.onnx',cfg.scene.num_envs)
                class HandoffActor:
                    def reset(self,ids=None):jump_actor.reset(ids);stand_actor.reset(ids)
                    def __call__(self,x):
                        weight=((env.episode_length_buf*.02-.22)/.14).clamp(0,1)
                        weight=weight.square()*(3-2*weight)
                        return torch.lerp(jump_actor(x),stand_actor(x),weight[:,None])
                actor=HandoffActor()
        elif a.source in ('trained','research'):
            policy_path=a.checkpoint or Path(selected['checkpoint'])
            wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
            runner=OnPolicyRunner(wrapped,c.to_dict(),log_dir=None,device=a.device)
            runner.load(str(policy_path),map_location=a.device)
            from tensordict import TensorDict
            native_actor=runner.get_inference_policy(device=a.device)
            if a.export_dir:
                from export_evidence import export_and_verify
                export_and_verify(runner,a.export_dir,policy_path,env._get_observations()['policy'])
            actor=lambda x:native_actor(TensorDict({"policy":x},batch_size=[cfg.scene.num_envs]),stochastic_output=a.stochastic)
        ui=None;hotkeys=None;suspended=[];events=[];flags={'reset':False,'pause':False,'nudge':None,'quit':False}
        if a.mode=='play':
            import carb.input,omni.appwindow,omni.ui as ui
            import omni.kit.viewport.utility as vu
            vp=vu.get_active_viewport()
            view_resolution=(960,720) if a.legacy_view else {'fast':(1280,720),'balanced':(1600,900),'quality':(1920,1080)}[a.view_profile]
            vp.set_texture_resolution(view_resolution)
            from omni.kit.hotkeys.core import get_hotkey_registry
            hotkeys=get_hotkey_registry()
            for h in hotkeys.get_all_hotkeys():
                combo=h.key_combination
                if combo and combo.modifiers==0 and str(getattr(combo.key,'name',combo.key)).upper() in ('SPACE','BACKSPACE','R','P','W','A','S','D','ESCAPE'):
                    if hotkeys.deregister_hotkey(h):suspended.append(h)
            inp=carb.input.acquire_input_interface();window=omni.appwindow.get_default_app_window()
            def key(event,*_):
                if event.type==carb.input.KeyboardEventType.KEY_PRESS:
                    name=str(getattr(event.input,'name',event.input)).upper()
                    if name in ('SPACE','BACKSPACE','R'):flags['reset']=True;events.append(name)
                    elif name=='P':flags['pause']=not flags['pause'];events.append(name)
                    elif name=='ESCAPE':flags['quit']=True;events.append(name)
                    elif name in ('W','A','S','D') and a.skill=='basketball':
                        flags['nudge']={'W':(.08,0),'S':(-.08,0),'A':(0,.08),'D':(0,-.08)}[name];events.append(name)
                return True
            subscription=inp.subscribe_to_keyboard_events(window.get_keyboard(),key)
            window.set_dpi_scale_override(1.0)
            if (getattr(a,'livestream',0)>0):
                # Keep browser rendering and the streamed window at the same size.
                # Restoring an OS window is inappropriate for headless WebRTC.
                window.resize(*view_resolution)
            else:
                window.restore_window();window.resize(1920,1200)
            def mouse_reset():flags.update(reset=True);events.append('MOUSE_RESET')
            def mouse_pause():flags.update(pause=not flags['pause']);events.append('MOUSE_PAUSE')
            panel=ui.Window('MicroDuck Isaac Skills',width=350,height=310 if a.skill=='basketball' else 230)
            with panel.frame:
                with ui.VStack(spacing=6):
                    ui.Label(a.skill+' / '+a.source+' / Newton MJWarp')
                    ui.Label('View: '+('legacy' if a.legacy_view else a.view_profile)+f' {view_resolution[0]}x{view_resolution[1]}   '+str(50 if a.legacy_view else a.view_fps)+' Hz')
                    ui.Label('Space/R: reset  P: pause  Esc: exit')
                    ui.Label('Alt + drag: orbit   Scroll: zoom')
                    ui.Button('Reset / Replay',clicked_fn=mouse_reset)
                    ui.Button('Pause / Resume',clicked_fn=mouse_pause)
                    if a.skill=='basketball':
                        ui.Label('WASD: nudge the free ball')
                        with ui.HStack():
                            ui.Button('Nudge left',clicked_fn=lambda:flags.update(nudge=(0,.08)))
                            ui.Button('Nudge right',clicked_fn=lambda:flags.update(nudge=(0,-.08)))
                    status=ui.Label('Normal speed: 1x target')
            panel.position_x=16 if (getattr(a,'livestream',0)>0) else 1500;panel.position_y=70
            for name in ('Content','Console','Simulation Output Settings','Property','Semantics Schema Editor','Stage','Layer','Render Settings'):
                old=ui.Workspace.get_window(name)
                if old:old.visible=False
            print('ISAAC_SKILLS_GUI_READY',flush=True)
        if a.video:
            import subprocess
            ffmpeg=ROOT/'work/isaac_tools/ffmpeg'
            a.video.parent.mkdir(parents=True,exist_ok=True)
            video_process=subprocess.Popen([str(ffmpeg),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','1280x720','-framerate','50','-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',str(a.video)],stdin=subprocess.PIPE)
            cam=env.scene.sensors['record_camera']
            eye=(.8,-.8,.62) if a.skill=='basketball' else ((.65,-.65,.38) if a.skill=='jump' else (.52,-.52,.30))
            look=(0,0,.30) if a.skill=='basketball' else (0,0,.16 if a.skill=='jump' else .12)
            from isaac_camera import aim_camera
            aim_camera(cam,eye,look)
            for _ in range(5):env.sim.render()
        pacer=None
        if a.mode=='play' and not a.legacy_view and not a.video:
            from isaac_viewport import RealtimeViewport
            import carb.settings
            # Kit's persisted 120 Hz cap sleeps inside every synchronous draw.
            # The viewport pacer already owns real-time timing.
            carb.settings.get_settings().set_bool('/app/runLoops/main/rateLimitEnabled',False)
            for _ in range(10):env.sim.render()
            print("ISAAC_VIEWPORT_WARMED",flush=True)
            pacer=RealtimeViewport(env,vp,a.view_fps)
        obs=env._get_observations()['policy']
        alive=torch.ones(cfg.scene.num_envs,dtype=torch.bool,device=a.device)
        first_failure=torch.full((cfg.scene.num_envs,),float('nan'),device=a.device)
        trial_active=alive.clone()
        integrity=None
        if a.mode=='eval':
            from physics_integrity import PhysicsIntegrity
            integrity=PhysicsIntegrity(env)
        histories={}; frames=[];frame_durations=[];paused_seconds=0.;start=time.monotonic();deadline=start;steps=0;resets=0
        pause_reference=None;pause_checks=[]
        def physical_snapshot():
            # Read Newton State directly; cached ArticulationData can hide UI-driven steps.
            import warp as wp
            manager=env.sim.backend_manager
            state=manager.get_state_0()
            values=[wp.to_torch(state.body_q),wp.to_torch(state.joint_q),wp.to_torch(state.joint_qd)]
            return np.concatenate([np.asarray(T(x).cpu() if hasattr(T(x),'cpu') else T(x)).ravel() for x in values]).copy()
        max_steps=int(a.duration/.02) if a.duration>0 else 2**50
        profiler=None
        if a.profile_python:
            import cProfile
            profiler=cProfile.Profile();profiler.enable()
        with torch.inference_mode():
            while (app is None or app.is_running()) and steps<max_steps and not flags['quit']:
                frame_start=time.monotonic()
                if a.mode=='play':
                    import omni.timeline
                    if flags['reset']:
                        env.reset();obs=env._get_observations()['policy'];flags['reset']=False;flags['pause']=False;resets+=1
                        if hasattr(actor,'reset'):actor.reset()
                        env.sim.set_camera_view(cfg.viewer.eye,cfg.viewer.lookat)
                        deadline=time.monotonic()
                        if pacer:pacer.rebase()
                    if flags['pause']:
                        status.text=f'{a.skill}   Paused | P: resume / Space: reset'
                        if a.verify_pause and pause_reference is None:
                            pause_reference=physical_snapshot()
                            pause_checks.append({'sim_seconds':steps*.02,'render_updates':0,'max_state_change':0.})
                        env.sim.render();time.sleep(.01);deadline=time.monotonic();paused_seconds+=deadline-frame_start
                        if a.verify_pause:
                            pause_checks[-1]['render_updates']+=1
                            pause_checks[-1]['max_state_change']=max(pause_checks[-1]['max_state_change'],float(np.max(np.abs(physical_snapshot()-pause_reference))))
                        if pacer:pacer.rebase()
                        continue
                    pause_reference=None
                if a.mode=='play' and flags['nudge'] is not None and env.ball is not None:
                    velocity=T(env.ball.data.root_vel_w).clone();velocity[:,:2]+=torch.tensor(flags['nudge'],device=a.device)
                    env.ball.write_root_velocity_to_sim_index(root_velocity=velocity);flags['nudge']=None
                if a.mode=='eval' and a.push_speed>0 and env.ball is not None and steps>0 and steps%100==0:
                    velocity=T(env.ball.data.root_vel_w).clone();angle=torch.rand(cfg.scene.num_envs,device=a.device)*2*np.pi
                    velocity[:,0]=a.push_speed*angle.cos();velocity[:,1]=a.push_speed*angle.sin()
                    env.ball.write_root_velocity_to_sim_index(root_velocity=velocity)
                actions=torch.zeros(cfg.scene.num_envs,14,device=a.device) if actor is None else actor(env.community_obs() if a.source=='community' else obs)
                if not torch.isfinite(actions).all():raise RuntimeError('Nonfinite policy output')
                if integrity is not None:integrity.active.copy_(trial_active)
                out=env.step(actions)
                obs=out[0]['policy'];terminated=out[2];truncated=out[3]
                if video_process is not None:
                    frame=T(cam.data.output['rgb'])[0].detach().cpu().numpy()[:,:,:3]
                    video_process.stdin.write(np.ascontiguousarray(frame,dtype=np.uint8).tobytes());video_frames+=1
                histories.setdefault('first_episode_active',[]).append(trial_active.detach().clone())
                new_failure=trial_active&terminated
                first_failure[new_failure]=steps*.02;alive&=~new_failure
                trial_active&=~(terminated|truncated)
                for k,v in env.rollout_metrics.items():
                    histories.setdefault(k,[]).append(v.detach().clone())
                if hasattr(actor,'reset'):actor.reset(torch.nonzero(terminated|truncated).flatten().cpu().numpy())
                steps+=1
                if a.mode=='play':
                    if pacer is None:
                        deadline+=.02;delay=deadline-time.monotonic()
                        if delay>0:time.sleep(delay)
                        elif delay<-.1:deadline=time.monotonic()
                    frames.append(time.monotonic());frame_durations.append(frames[-1]-frame_start)
                    if steps==5:
                        if (getattr(a,'livestream',0)>0):
                            window.resize(*view_resolution)
                        else:
                            window.restore_window();window.resize(1920,1200)
                    if steps==10:
                        panel.visible=True
                        if not (getattr(a,'livestream',0)>0):panel.focus()
                        panel.position_x=16 if (getattr(a,'livestream',0)>0) else 1500
                        panel.position_y=70
                    if steps%25==0:
                        fps=pacer.recent_fps() if pacer else 1/np.mean(frame_durations[-50:])
                        status.text=f'{a.skill}   {.02/np.mean(frame_durations[-50:]):.2f}x   View: {fps:.1f} fps'
                    if steps==50:
                        import carb.settings
                        st=carb.settings.get_settings()
                        view_info={'profile':a.view_profile,'texture_resolution':list(vp.get_texture_resolution()),'physics_backend':type(env.sim.backend_manager).__module__+'.'+type(env.sim.backend_manager).__qualname__,'settings':{k:st.get(k) for k in ['/rtx/post/aa/op','/rtx/post/dlss/execMode','/rtx-transient/dldenoiser/enabled','/app/asyncRendering','/app/runLoops/main/rateLimitEnabled','/app/runLoops/main/rateLimitFrequency','/app/runLoops/present/rateLimitEnabled','/app/runLoops/rendering_0/rateLimitEnabled','/app/renderer/sleepMsOnFocus','/app/renderer/sleepMsOutOfFocus','/rtx/ecoMode/enabled','/rtx/reflections/enabled','/rtx/indirectDiffuse/enabled','/rtx/shadows/enabled','/rtx/ambientOcclusion/enabled']}}
                        print('ISAAC_VIEWPORT_INFO='+json.dumps(view_info,default=str),flush=True)
        if profiler:
            profiler.disable();a.profile_python.parent.mkdir(parents=True,exist_ok=True);profiler.dump_stats(str(a.profile_python))
        if video_process is not None:
            video_process.stdin.close()
            if video_process.wait(timeout=30)!=0:raise RuntimeError("Video encoder failed")
            video_process=None
        data={k:torch.stack(v).cpu().numpy() for k,v in histories.items()}
        summary=dict(skill=a.skill,robot_variant=a.robot_variant,robot_variant_selection=variant_reason,source=a.source,mode=a.mode,backflip_recovery_curriculum=cfg.backflip_recovery_curriculum,backflip_strict_finish=cfg.backflip_strict_finish,prime_reset_delay=cfg.prime_reset_delay,basketball_recipe=recipe,stochastic=a.stochastic,handoff=a.handoff,root_height=cfg.root_height,disable_sleep=a.awake,push_speed=a.push_speed,reset_noise=a.reset_noise,backend='Isaac Lab develop 5088b326 / Newton MJWarp native BAM',physics=backend_evidence(env.sim),domain_randomization=env.dr_evidence,actuator=a.actuator,seed=a.seed,
            num_envs=cfg.scene.num_envs,steps=steps,sim_seconds=steps*.02,wall_seconds=time.monotonic()-start,
            policy=str(policy_path),policy_sha256=sha(policy_path) if policy_path else None,
            survival_rate=alive.float().mean().item(),first_failure_seconds=[None if not np.isfinite(x) else float(x) for x in first_failure.cpu().numpy()],
            paused_wall_seconds=paused_seconds,active_wall_seconds=time.monotonic()-start-paused_seconds,manual_resets=resets,input_events=events,video=str(a.video) if a.video else None,video_frames=video_frames,video_fps=50 if a.video else None,metrics={k:dict(mean=float(v.mean()),min=float(v.min()),max=float(v.max())) for k,v in data.items()})
        if a.skill=='backflip' and a.source=='community':
            summary['transfer_scope']='Published v88 assisted cube launch actor in unassisted flat-ground Newton test; terrain, spotter and v44 recovery handoff intentionally absent; not a reproduction of its assisted success claim.'
        if a.mode=='play':
            summary['viewport']=view_info if steps>=50 else {}
            if a.verify_pause:summary['pause_checks']=pause_checks
        if prop_info:summary['community_prop']=prop_info
        if pacer:
            summary['viewport_timing']=pacer.summary();pacer.close()
        if frames:
            delta=np.asarray(frame_durations)*1000;summary['frames_ms']={f'p{q}':float(np.percentile(delta,q)) for q in [50,95,99]}
            summary['frames_ms_definition']='Policy-loop duration including pacing. Use viewport_timing for display update intervals.'
        active=data['first_episode_active']
        summary['first_episode_metrics']={k:dict(mean=float(v[active].mean()),min=float(v[active].min()),max=float(v[active].max())) for k,v in data.items() if k!='first_episode_active'}
        if a.skill=='backflip':
            succeeded=np.where(active,data['backflip_success'],0).max(axis=0)>0
            trial_success=succeeded&alive.cpu().numpy()
            rotation=np.where(active,data['airborne_rotation_deg'],0).max(axis=0)
            launches=np.where(active,data['flip_takeoff'],0).max(axis=0)>0
            feet_first=np.where(active,data['feet_first'],0).max(axis=0)>0
            stable_hold=np.where(active,data['stable_landing_seconds'],0).max(axis=0)
            summary['trial_backflip_metrics']=dict(rotation_degrees=rotation.tolist(),takeoff=launches.tolist(),feet_first=feet_first.tolist(),stable_hold=stable_hold.tolist())
            contact_ids=np.where(active,data['first_contact_body_index'],-1).max(axis=0).astype(int)
            contact_names=list(env.body_contacts.sensor_names or [])
            summary['first_contact_body_counts']={('none' if i<0 else contact_names[i] if i<len(contact_names) else str(i)):int((contact_ids==i).sum()) for i in np.unique(contact_ids)}
            summary.update(backflip_takeoff_fraction=float(launches.mean()),backflip_full_rotation_fraction=float((rotation>=360).mean()),backflip_feet_first_fraction=float(feet_first.mean()),backflip_rotation_mean_degrees=float(rotation.mean()))
            summary['backflip_success_fraction']=float((succeeded&alive.cpu().numpy()).mean())
            summary['task_success']=bool(summary['backflip_success_fraction']>=.9)
            summary['criterion']='Standing start, no assistance, all-body collision-free flight with >=360 deg backward rotation before first recontact; feet-first and stable >=0.5 s in >=90% trials'
        elif a.skill=='arabesque':
            held=np.where(active,data['pose_hold_seconds'],0).max(axis=0)
            trial_success=(held>=4)&alive.cpu().numpy()
            summary['arabesque_success_fraction']=float(((held>=4)&alive.cpu().numpy()).mean())
            summary['task_success']=bool(summary['arabesque_success_fraction']>=.9)
            summary['criterion']='>=4 s continuous left-foot support with right sole 35-80 mm above ground, >=10 mm behind left foot, tilt <0.35 rad, drift <0.15 m in >=90% trials'
        elif a.skill=='dance':
            trial_yaw=np.where(active,data['yaw_rms'],0).sum(axis=0)/active.sum(axis=0).clip(min=1)
            trial_drift=np.where(active,data['drift'],0).max(axis=0)
            trial_success=alive.cpu().numpy()&(trial_yaw<.15)&(trial_drift<.15)
            summary['dance_success_fraction']=float(trial_success.mean())
            summary['task_success']=bool(summary['survival_rate']>=.9 and data['yaw_rms'][active].mean()<.15 and data['drift'][active].max()<.15)
            summary['criterion']='>=90% survival, hip-yaw RMS <0.15 rad, drift <0.15 m'
        elif a.skill=='basketball':
            trial_success=alive.cpu().numpy()&(a.duration>=10)
            summary['task_success']=bool(summary['survival_rate']>=.9 and a.duration>=10)
            summary['criterion']='>=90% first-episode survival for at least 10 s on a free ball'
        else:
            peaks=np.where(active,data['visible_jump'],0).max(axis=0)
            airs=np.where(active,data['qualified_airtime'],0).max(axis=0)
            land=np.where(active,data['stable'],0).max(axis=0)>0
            trial_success=(peaks>=.012)&(airs>=.08)&land&alive.cpu().numpy()
            summary['trial_jump_metrics']=dict(visible_height=peaks.tolist(),airtime=airs.tolist(),stable_landing=land.tolist())
            summary['jump_success_fraction']=float(((peaks>=.012)&(airs>=.08)&land&alive.cpu().numpy()).mean())
            summary['task_success']=bool(summary['jump_success_fraction']>=.9)
            summary['criterion']='>=12 mm min(CoM rise since takeoff, bilateral sole clearance), >=80 ms supported launch/flight, stable two-foot landing for 1 s in >=90% trials'
        summary.update(success_fraction=float(trial_success.mean()),success_count=int(trial_success.sum()),trial_success=trial_success.tolist(),success_threshold=a.success_threshold)
        summary['task_success']=bool(summary['success_fraction']>=a.success_threshold)
        summary['criterion']=summary['criterion'].replace('90%',f'{100*a.success_threshold:g}%')
        if integrity is not None:
            # Preserve the historical metrics; frozen release qualification is
            # stricter and independently includes per-physics-step integrity.
            physical=integrity.report();qualified=trial_success&np.asarray(physical['passed_per_trial'])
            required_duration=3.5 if a.skill=='jump' else 30. if a.skill in ('basketball','dance') else a.duration
            qualified&=steps*.02+1e-8>=required_duration
            extra={}
            if a.skill=='dance':
                rmse=np.sqrt(np.where(active,np.square(data['yaw_rms']),0).sum(0)/active.sum(0).clip(min=1))
                qualified&=rmse<.15
                extra['hip_yaw_tracking_rmse_rad']=rmse.tolist()
            frozen={'protocol':{'basketball':'basketball-free-v1','jump':'jump-basic-v1','dance':'dance-basic-v1'}.get(a.skill,'challenge'),
                'physical_integrity':physical,'required_duration_s':required_duration,
                'actual_duration_s':steps*.02,'required_trials_per_seed':256,'required_seed_groups':3,
                'minimum_group_success_rate':.95,'success_count':int(qualified.sum()),'trial_success':qualified.tolist(),
                'success_fraction':float(qualified.mean()),'group_passed':bool(len(qualified)==256 and qualified.mean()>=.95),
                'release_passed':False,'release_note':'One group only; three distinct held-out groups and both model lines require external aggregation.',**extra}
            summary.update(legacy_task_success=summary['task_success'],qualification=frozen)
            summary['task_success']=bool(frozen['group_passed'])
        if hotkeys is not None:
            for h in suspended:hotkeys.register_hotkey(h)
            inp.unsubscribe_to_keyboard_events(window.get_keyboard(),subscription)
        env.dr_evidence['actual_final_parameters']=observed_randomization(env,'after evaluation/play physical steps')
        summary['evaluation_wall_seconds']=time.monotonic()-start
        summary['startup_wall_seconds']=start-startup_started
        outpath=a.output or OUT/'evaluations' /f'{a.skill}_{a.source}_seed{a.seed}_{a.actuator}.json'
        outpath.parent.mkdir(parents=True,exist_ok=True);outpath.write_text(json.dumps(summary,indent=2)+'\n')
        np.savez_compressed(outpath.with_suffix('.npz'),**data)
        print('ISAAC_SKILLS_RESULT='+json.dumps(summary),flush=True)
    except BaseException as exc:
        if accounting is not None:accounting.finish(error=exc)
        raise
    finally:
        if training_video:training_video.finish()
        if video_process is not None:
            video_process.stdin.close();video_process.wait(timeout=30)
        if env is not None:env.close()
        if runtime_context is not None:runtime_context.__exit__(*sys.exc_info())
try:
    main()
    if a.completion_file:a.completion_file.write_text("complete\n")
except BaseException:
    import traceback
    traceback.print_exc()
    raise
