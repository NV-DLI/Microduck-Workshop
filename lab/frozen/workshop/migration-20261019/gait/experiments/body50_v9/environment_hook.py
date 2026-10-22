"""Real BODY50 observations and a temporary physical fixture for RL resets.

Actor inputs are synthesized from the source sensor pipeline. Privileged state
is used only by that producer, the startup fixture, rewards and the critic.
"""
import torch
from startup_fixture import StartupFixture
from physics_producer import Body50PhysicsProducer
from vector_body_imu import VectorBodyProfile, rotate


def T(value):
    value = getattr(value, 'torch', value)
    if isinstance(value, torch.Tensor):
        return value
    import warp as wp
    return wp.to_torch(value)


def startup_failed(env):
    hook = getattr(env, 'body50_hook', None)
    return hook.fixture.failed if hook else torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


class AdditionalObservationDR:
    """Declared upstream mounting/noise/delay robustness on top of BODY50."""
    def __init__(self, n, device):
        axis = torch.nn.functional.normalize(torch.randn(n, 3, device=device), dim=-1)
        angle = torch.rand(n, 1, device=device) * torch.pi/30
        self.mount = torch.cat((axis*torch.sin(angle/2), torch.cos(angle/2)), -1)
        self.lags = torch.zeros((n, 2), device=device, dtype=torch.long)
        self.phase = torch.randint(0, 64, (n, 2), device=device)
        self.previous = torch.zeros((n, 6), device=device)
        self.previous_qd = torch.zeros((n, 14), device=device)
        self.cache = torch.zeros((n, 48), device=device)
        self.valid = torch.zeros(n, dtype=torch.bool, device=device)
        self.last_step = -1

    def reset(self, ids):
        ids = slice(None) if ids is None else ids
        self.previous[ids] = 0
        self.previous_qd[ids] = 0
        self.cache[ids] = 0
        self.valid[ids] = False

    def apply(self, obs, ready, step):
        if self.last_step == step:
            out = obs.clone()
            out[:, :48] = torch.where(self.valid[:, None], self.cache, obs[:, :48])
            return out
        self.last_step = step
        out = obs.clone()
        readings = torch.cat((rotate(self.mount, obs[:, :3]), rotate(self.mount, obs[:, 3:6])), -1)
        update = (step + self.phase) % 64 == 0
        self.lags = torch.where(update, torch.randint(0, 2, self.lags.shape, device=obs.device), self.lags)
        out[:, :3] = torch.where((self.lags[:, :1] > 0) & self.valid[:, None], self.previous[:, :3], readings[:, :3])
        out[:, 3:6] = torch.where((self.lags[:, 1:] > 0) & self.valid[:, None], self.previous[:, 3:6], readings[:, 3:6])
        out[:, :3] += (torch.rand_like(out[:, :3])*2-1)*.03
        out[:, 3:6] += (torch.rand_like(out[:, 3:6])*2-1)*.01
        out[:, 6:20] += (torch.rand_like(out[:, 6:20])*2-1)*.001
        out[:, 20:34] = torch.where(self.valid[:, None], self.previous_qd, obs[:, 20:34])
        out[:, 20:34] += (torch.rand_like(out[:, 20:34])*2-1)*.25
        out = torch.where(ready[:, None], out, torch.zeros_like(out))
        self.previous.copy_(readings)
        self.previous_qd.copy_(obs[:, 20:34])
        self.cache.copy_(out[:, :48])
        self.valid.copy_(ready)
        return out


class Body50TrainingHook:
    def __init__(self, env, *, seed, extra_observation_dr=True):
        from bounded_actions import JOINT_NAMES
        self.env, self.robot = env, env.scene['robot']
        self.ids, names = self.robot.find_joints(JOINT_NAMES, preserve_order=True)
        self.pids, _ = self.robot.find_joints(['passive_'+n+'_backlash' for n in JOINT_NAMES], preserve_order=True)
        self.body_ids, _ = self.robot.find_bodies(['trunk_base'], preserve_order=True)
        if list(names) != JOINT_NAMES or self.robot.num_joints != 28 or len(self.body_ids) != 1:
            raise ValueError('Native28-joint robot with exact14 action order required')
        self.trunk = self.body_ids[0]
        self.producer = Body50PhysicsProducer(env.num_envs, device=env.device,
            profile=VectorBodyProfile(orientation_correction_tau_s=2.), seed=seed+1)
        self.fixture = StartupFixture(env.num_envs, env.device)
        self.robustness = AdditionalObservationDR(env.num_envs, env.device) if extra_observation_dr else None
        self.previous_raw = torch.zeros((env.num_envs, 14), device=env.device)
        self.task_start_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.task_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.initialization_steps = self.task_steps.clone()
        self.startup_failures = self.task_steps.clone()
        self.physics_calls = 0
        self.task_wrench_peak = torch.zeros((), device=env.device)
        self._last_physics = env.sim.get_physics_step_count()
        env.body50_hook = self
        env.ready_mask = self.fixture.ready
        env.task_active_mask = self.fixture.task_active
        env.sim.backend_manager.set_decimation(1)
        env._physics_handles_decimation = False
        self._install()
        self.reset(None)

    def reset(self, ids):
        ids = slice(None) if ids is None else ids
        self.producer.reset(ids)
        self.previous_raw[ids] = 0
        if self.robustness:
            self.robustness.reset(ids)
        data = self.robot.data
        reference = torch.cat((T(data.root_pos_w), T(data.root_quat_w)), -1).clone()
        self.fixture.reset(ids, reference_pose_xyzw=reference, actual_mass_kg=T(data.body_mass).sum(-1))
        self._sync_masks()

    def _sync_masks(self):
        self.env.ready_mask = self.fixture.ready
        self.env.task_active_mask = self.fixture.task_active

    def _install(self):
        env = self.env
        reset = env._reset_idx
        def reset_idx(ids):
            self.startup_failures[ids] += self.fixture.failed[ids].to(torch.long)
            reset(ids)
            self.reset(ids)
        env._reset_idx = reset_idx

        write = env.scene.write_data_to_sim
        def write_data():
            data = self.robot.data
            ready = self.producer.observe()['orientation_valid']
            force, torque = self.fixture.update(self.producer.ticks*.005,
                position=T(data.root_pos_w), quaternion=T(data.root_quat_w),
                linear_velocity=T(data.root_lin_vel_w), angular_velocity=T(data.root_ang_vel_w), orientation_valid=ready)
            self._sync_masks()
            self.robot.permanent_wrench_composer.set_forces_and_torques_index(
                forces=force[:, None], torques=torque[:, None], body_ids=self.body_ids, is_global=True)
            write()
            actual = T(self.robot.data._sim_bind_body_external_wrench)
            # All task worlds must have exactly zero assistance in the actual buffer.
            actual = actual.reshape(env.num_envs, -1)
            self.task_wrench_peak = torch.maximum(self.task_wrench_peak,
                (actual.abs()*self.fixture.task_active[:, None]).amax())
        env.scene.write_data_to_sim = write_data

        update = env.scene.update
        def scene_update(dt):
            update(dt)
            tick = env.sim.get_physics_step_count()
            if tick == self._last_physics:
                return
            if tick != self._last_physics+1 or abs(dt-.005)>1e-9:
                raise RuntimeError('BODY50 requires every real200Hz physics step')
            self._last_physics = tick
            self.physics_calls += 1
            data = self.robot.data
            term = env.action_manager.get_term('joint_pos')
            self.producer.update(trunk_position_world_m=T(data.body_pos_w)[:, self.trunk],
                trunk_quaternion_xyzw=T(data.body_quat_w)[:, self.trunk],
                trunk_angular_velocity_world_rad_s=T(data.body_ang_vel_w)[:, self.trunk],
                effective_encoder_q_rad=T(data.joint_pos)[:, self.ids]+T(data.joint_pos)[:, self.pids]+term.encoder_bias,
                effective_encoder_qd_rad_s=T(data.joint_vel)[:, self.ids]+T(data.joint_vel)[:, self.pids])
            self.fixture.ready = self.producer.observe()['orientation_valid'] & ~self.fixture.failed
            self._sync_masks()
        env.scene.update = scene_update

        process = env.action_manager.process_action
        def process_action(action):
            effective = torch.where(env.ready_mask[:, None], action, torch.zeros_like(action))
            process(effective)
            self.previous_raw.copy_(effective)
        env.action_manager.process_action = process_action

        commands = env.command_manager.compute
        def compute_commands(dt):
            commands(dt)
            for key in ('base_velocity', 'head_pose', 'body_pose'):
                value = env.command_manager.get_command(key)
                value[~env.task_active_mask] = 0
        env.command_manager.compute = compute_commands

        observations = env.observation_manager.compute
        def compute_observations(*args, **kwargs):
            result = observations(*args, **kwargs)
            if 'policy' in result:
                obs, ready = self.producer.build_actor_observation(result['policy'],
                    native_home14=T(self.robot.data.default_joint_pos)[:, self.ids], previous_raw_action14=self.previous_raw)
                if self.robustness:
                    obs = self.robustness.apply(obs, ready, env.common_step_counter)
                result['policy'] = obs
            return result
        env.observation_manager.compute = compute_observations

        rewards = env.reward_manager.compute
        def compute_reward(dt):
            reward = rewards(dt)
            rm = env.reward_manager
            invalid = ~self.task_start_mask
            rm._episode_sum_buf.sub_(rm._step_reward*invalid[:, None]*dt)
            rm._step_reward[invalid] = 0
            reward[invalid] = 0
            return reward
        env.reward_manager.compute = compute_reward

        step = env.step
        def step_env(action):
            self.task_start_mask.copy_(env.task_active_mask)
            result = step(action)
            _, _, terminated, timeout, extras = result
            inactive = ~self.task_start_mask
            env.episode_length_buf.sub_((inactive & ~(terminated|timeout) & (env.episode_length_buf>0)).to(torch.long))
            self.task_steps.add_(self.task_start_mask.to(torch.long))
            self.initialization_steps.add_(inactive.to(torch.long))
            extras['body50_task_start_mask'] = self.task_start_mask.clone()
            extras['body50_next_ready_mask'] = env.ready_mask.clone()
            return result
        env.step = step_env

    def summary(self):
        return {'schema':'microduck.body50-training-hook.v1', 'physics_calls':self.physics_calls,
            'ready_environments':int(self.env.ready_mask.sum()), 'task_environments':int(self.env.task_active_mask.sum()),
            'task_env_steps':int(self.task_steps.sum()), 'initialization_env_steps_excluded':int(self.initialization_steps.sum()),
            'startup_failures':int(self.startup_failures.sum()), 'actual_task_external_wrench_peak':float(self.task_wrench_peak),
            'additional_official_observation_DR':self.robustness is not None,
            'physical_state_projection':False, 'native_physics_decimation':1,
            'policy_hz':50, 'physics_hz':200, 'actor_dimensions':61,
            'source_contract':'BODY50 median3/25-read gate/tau2, same-read quantized encoders, raw action history',
            'privileged_scope':'virtual sensor, initialization fixture, critic and reward only'}
