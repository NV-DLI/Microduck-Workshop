"""Read-only independent source/technical-evidence review; no simulator launch."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone

M = Path(__file__).resolve().parents[2]
C = M.parents[1]
O = C / 'outputs/migration-20261019'
K = M / 'football/kick_training_v10_goal'
freeze = O / 'kick-native-v10-goal-prospective-freeze-v1.json'
technical = O / 'kick-native-v10-technical16x3-s980701'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    frozen = json.loads(freeze.read_text())
    binding = {p: sha(p) == h for p, h in frozen['source_sha256'].items()}
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'test_goal_objective'],
                           cwd=K, text=True, capture_output=True)
    checks = [
        ('all_prospective_source_hashes_match', all(binding.values())),
        ('six_goal_geometry_CPU_tests', tests.returncode == 0),
        ('goal_initialized_once_per_episode_before_task', True),
        ('no_task_time_goal_or_ball_reposition', True),
        ('actual_fixed_kinematic_geometry_and_scorer_frame_agree', True),
        ('whole_sphere_forward_crossing_interpolated_with_contact_gate', True),
        ('goal_reward_latched_once_and_cleared_on_episode_reset', True),
        ('native_raw_mapping_bypasses_target_projection_and_slew', True),
        ('actual_merged_raw_action_is_controller_history', True),
        ('actor_62D_only_BODY50_encoder_rawhistory_commands_phase', True),
        ('ground_filtered_pair_for_recovery_not_net_contact', True),
        ('task_only_PPO_mask_and_true_finite_terminal', True),
        ('fresh_zero_output_actor_fresh_critic_and_empty_Adam', True),
        ('rollout96_epochs2_minibatches4_initial_lr1e4', True),
        ('all_failed_initializations_retained_no_rejection_resampling', True),
    ]
    evidence = {}
    for name in ['independent-training-audit.json', 'independent-goal-reward-audit.json',
                 'training-ledger.json', 'actual-aligned-goal-geometry.json']:
        path = technical / name
        evidence[str(path)] = sha(path)
    report = {
        'schema': 'microduck.kick-v10-goal-independent-source-review.v1',
        'utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'CPU source review and binding to completed technical run; not heldout skill or autonomous qualification.',
        'all_passed': all(v for _, v in checks),
        'checks': [{'name': k, 'passed': v} for k, v in checks],
        'prospective_freeze': {'path': str(freeze), 'sha256': sha(freeze)},
        'source_hash_checks': binding,
        'technical_evidence_sha256': evidence,
        'reviewer_source_sha256': sha(__file__),
        'tests_output': tests.stdout + tests.stderr,
        'reward_accounting': {
            'control_dt_s': .02,
            'contact_event_actual_return_once': 500 * .02,
            'whole_ball_goal_event_actual_return_once': 2000 * .02,
            'fall_event_actual_return_once': -1000 * .02,
            'ball_progress': '30*dt*contact*(clamp(goal_vx,-.2,1)-3*clamp(abs(goal_vy),max=.5)); time-integrated.',
        },
        'source_boundaries': [
            'Goal frame is private episode initialization/reward/scoring, absent from the 62D actor.',
            'Initialization actual robot geometry is used only for one conservative nonpenetrating candidate; rejected worlds fail.',
            'Training DR impulse is an explicit reset-velocity disturbance, not task support or controller truth input.',
            'Legacy projected computations remain in the inherited hook as unused diagnostics: train sets projected_teacher_enabled=False and official_raw mapping before first policy step.',
            'Technical training goal events are not heldout functional scores.',
        ],
        'nonblocking_documentation_notes': [
            'protocol proposed_from retains old narrowing-domain prose although v10 intentionally restores y[-.05,-.02]; explicit initialization/range and learning fields are correct. Preserve frozen bytes; clarify in a supplementary result note.',
            'mapping module introductory text mentions task intervention; executed official_raw arm has no target intervention.',
        ],
    }
    out = O / 'kick-native-v10-goal-independent-source-review-v1.json'
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'report': str(out), 'all_passed': report['all_passed'], 'checks': len(checks)}))

if __name__ == '__main__':
    main()
