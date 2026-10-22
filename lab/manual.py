"""Keyboard football with the two Pollen Robotics ONNX policies."""
from pathlib import Path
import argparse,hashlib,json,sys,time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--models',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=600)
    p.add_argument('--headless-check',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.runtime/'gait'))
    import torch,gymnasium as gym
    from isaaclab.app import launch_simulation
    from batch_policy import BatchPolicy
    with launch_simulation(None,{'device':'cuda:0','require_kit':True,'visualizer':[] if a.headless_check else ['kit']}):
        import isaaclab_tasks,isaaclab.sim as sim_utils
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab.assets import AssetBaseCfg,RigidObjectCfg
        from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg
        cfg=parse_env_cfg('IsaacContrib-Velocity-Flat-MicroDuck',device='cuda:0',num_envs=1,overrides=['presets=backlash'])
        if not a.headless_check:
            from isaaclab_visualizers.kit import KitVisualizerCfg
            cfg.sim.visualizer_cfgs=[KitVisualizerCfg()]
        cfg.seed=42;cfg.curriculum=None;cfg.episode_length_s=a.seconds+10
        cfg.scene.robot.spawn.usd_path=str(a.runtime/'visuals/assets/microduck_white_orange.usda')
        cfg.scene.robot.init_state.pos=(-.66,0.,.125)
        cfg.scene.ball=RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Ball',spawn=sim_utils.SphereCfg(radius=.035,rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),mass_props=sim_utils.MassPropertiesCfg(mass=.045),collision_props=sim_utils.CollisionPropertiesCfg(),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.9,.91,.88),roughness=.75)),init_state=RigidObjectCfg.InitialStateCfg(pos=(0.,0.,.035)))
        for name,size,pos in [('goal_left',(.025,.025,.1925),(.93,.1325,.09625)),('goal_right',(.025,.025,.1925),(.93,-.1325,.09625)),('goal_bar',(.025,.29,.025),(.93,0.,.1925))]:
            setattr(cfg.scene,name,AssetBaseCfg(prim_path='{ENV_REGEX_NS}/'+name,spawn=sim_utils.CuboidCfg(size=size,collision_props=sim_utils.CollisionPropertiesCfg(),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.9,.92,.88),roughness=.7)),init_state=AssetBaseCfg.InitialStateCfg(pos=pos)))
        cfg.scene.pitch=AssetBaseCfg(prim_path='{ENV_REGEX_NS}/PitchVisual',spawn=sim_utils.CuboidCfg(size=(2.,1.,.003),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.12,.32,.16),roughness=.85)),init_state=AssetBaseCfg.InitialStateCfg(pos=(0.,0.,-.0013)))
        cfg.scene.sky_light.spawn.intensity=1800.
        cfg.sim.physics.solver_cfg.nconmax=200;cfg.sim.physics.solver_cfg.njmax=1024
        cfg.commands.base_velocity.resampling_time_range=(1e8,1e8);cfg.commands.base_velocity.debug_vis=False
        cfg.commands.head_pose.ranges=tuple((0.,0.)for _ in range(4));cfg.commands.body_pose.ranges=tuple((0.,0.)for _ in range(6));cfg.events.reset_base.params['pose_range']={}
        for name in ['foot_friction','encoder_bias','mass_inertia','randomize_com','randomize_head_com','randomize_joint_friction','randomize_armature','push_robot']:setattr(cfg.events,name,None)
        cfg.scene.robot.actuators['servos'].vin_range=None;cfg.scene.robot.actuators['servos'].vin_drop_gain_range=None;cfg.scene.robot.actuators['servos'].min_delay=4;cfg.scene.robot.actuators['servos'].max_delay=4
        cfg.observations.policy.enable_corruption=False;cfg.observations.policy.imu.params.update(max_angle_deg=0,min_lag=0,max_lag=0)
        env=gym.make('IsaacContrib-Velocity-Flat-MicroDuck',cfg=cfg).unwrapped;env.reset();env.sim.set_camera_view((1.35,-1.45,.9),(.05,0,.1))
        manager=env.sim.backend_manager;assert manager.backend_name=='newton' and cfg.sim.use_newton_actuators
        manager.set_decimation(1);env._physics_handles_decimation=False
        hashes={f:hashlib.sha256((a.models/f).read_bytes()).hexdigest()for f in ['walk.onnx','kick-teacher.onnx']}
        walk=BatchPolicy(a.models/'walk.onnx',hashes['walk.onnx'],'cuda:0');kick=BatchPolicy(a.models/'kick-teacher.onnx',hashes['kick-teacher.onnx'],'cuda:0')
        held=set();flags=dict(reset=False,kick=False,quit=False);events=[];kick_start=None;fallen=False;scored=False;resets=0
        if not a.headless_check:
            import carb.input,omni.appwindow,omni.ui as ui
            import omni.kit.viewport.utility as vu
            from omni.kit.hotkeys.core import get_hotkey_registry
            registry=get_hotkey_registry();suspended=[]
            for hotkey in registry.get_all_hotkeys():
                combo=hotkey.key_combination
                if combo and combo.modifiers==0 and str(getattr(combo.key,'name',combo.key)).upper()in ('W','A','S','D','Q','E','SPACE','R','ESCAPE'):
                    if registry.deregister_hotkey(hotkey):suspended.append(hotkey)
            inp=carb.input.acquire_input_interface();window=omni.appwindow.get_default_app_window()
            def key(event,*_):
                name=str(getattr(event.input,'name',event.input)).upper()
                if event.type==carb.input.KeyboardEventType.KEY_PRESS:
                    held.add(name)
                    if name=='R':flags['reset']=True
                    elif name=='SPACE':flags['kick']=True
                    elif name=='ESCAPE':flags['quit']=True
                    events.append({'key':name,'at':time.monotonic()})
                elif event.type==carb.input.KeyboardEventType.KEY_RELEASE:held.discard(name)
                return True
            subscription=inp.subscribe_to_keyboard_events(window.get_keyboard(),key)
            window.resize(1920,1080);vu.get_active_viewport().set_texture_resolution((1920,1080))
            panel=ui.Window('Microduck • Keyboard football',width=390,height=235)
            with panel.frame:
                with ui.VStack(spacing=8):
                    ui.Label('W/S: forward/back   A/D: step sideways')
                    ui.Label('Q/E: turn   Space: right-foot kick')
                    ui.Label('Release keys to stand. R: restart. Esc: exit.')
                    ui.Label('After restart, release and press movement keys again.')
                    ui.Label('Official walking + kicking ONNX policies')
                    ui.Button('Restart scene',clicked_fn=lambda:flags.update(reset=True))
                    status=ui.Label('Approach the ball; stop before kicking.')
            print('MANUAL_FOOTBALL_READY',flush=True)
        cmd=env.command_manager.get_term('base_velocity');robot=env.scene['robot'];rows=[];start=time.monotonic()
        def T(x):return getattr(x,'torch',x)
        with torch.inference_mode():
            for step in range(round(a.seconds/.02)):
                tick_start=time.monotonic()
                if flags['quit']:break
                if flags['reset']:
                    env.reset();flags.update(reset=False,kick=False);held.clear();kick_start=None;fallen=False;scored=False;resets+=1
                if a.headless_check and step==150:flags['kick']=True
                if flags['kick'] and kick_start is None and not fallen:kick_start=step;flags['kick']=False
                cmd.is_standing_env[:]=False;cmd.vel_command_b[:]=0
                if kick_start is None and not fallen:
                    cmd.vel_command_b[0]=torch.tensor([.25*('W'in held)-.18*('S'in held),.15*('A'in held)-.15*('D'in held),.45*('Q'in held)-.45*('E'in held)],device='cuda:0')
                env.command_manager.get_term('head_pose').command[:]=0;env.command_manager.get_term('body_pose').command[:]=0
                obs=env.observation_manager.compute(update_history=True)['policy'];action=walk(obs)
                if kick_start is not None:
                    phase=(step-kick_start)*.02
                    # A bounded official kick, then a smooth handoff to standing.
                    weight=1. if phase<.14 else max(0.,1.-(phase-.14)/.12)
                    if weight:action=torch.lerp(action,kick(obs),weight)
                    if phase>=1.:kick_start=None;flags['kick']=False
                env.action_manager.process_action(action)
                for tick in range(4):
                    env._sim_step_counter+=1;env.action_manager.apply_action();env.scene.write_data_to_sim();env.sim.step(render=tick==3);env.scene.update(.005)
                env.episode_length_buf+=1;env.common_step_counter+=1
                fallen=fallen or bool((T(robot.data.projected_gravity_b)[0,2]>-.6)|(T(robot.data.root_pos_w)[0,2]<.065))
                ball=T(env.scene['ball'].data.root_pos_w)[0]
                scored=scored or bool(ball[0]>.9775 and abs(ball[1])<.085 and ball[2]<.145)
                if step%5==0:rows.append(dict(t=(step+1)*.02,position=T(robot.data.root_pos_w)[0].cpu().tolist(),ball=ball.cpu().tolist(),fell=fallen,kicking=kick_start is not None,command=cmd.vel_command_b[0].cpu().tolist()))
                if not a.headless_check:
                    status.text='Fallen — press R to restart.'if fallen else ('Goal! Press R to try again.'if scored else 'Position the ball beside the right foot; release keys; Space.')
                    time.sleep(max(0,.02-(time.monotonic()-tick_start)))
        if not a.headless_check:inp.unsubscribe_to_keyboard_events(window.get_keyboard(),subscription)
        report=dict(status='completed',models=hashes,physics_backend=manager.backend_name,physics_hz=200,policy_hz=50,resets=resets,fell=fallen,goal_observed=scored,events=events,scope='keyboard skill exploration; goal position observation is not autonomous-task certification')
        (a.output/'report.json').write_text(json.dumps(report,indent=2));(a.output/'trajectory.json').write_text(json.dumps(rows));env.close()
if __name__=='__main__':main()
