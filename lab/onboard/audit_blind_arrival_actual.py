"""Independent recorded-event audit; does not import the production blind gate."""
import bisect
import math


def number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def vector(x, n):
    return isinstance(x, list) and len(x) == n and all(number(v) for v in x)


def eq(a, b):
    if number(a) and number(b):
        return abs(a-b) <= 1e-8
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
    return a == b


def zero(x, n):
    return vector(x, n) and max(abs(v) for v in x) <= 1e-9


def fresh(t, stamp, limit):
    return number(stamp) and 0 <= t-stamp <= limit+1e-9


def domain(p):
    return vector(p, 3) and .07 <= p[0] <= .15 and -.10 <= p[1] <= .01


def common(r, trace, *, exit=False):
    t = r['timestamp_s']
    h = r['public_handoff']
    m = h['memory']
    b = r['blind_arrival']
    c = b['capture_context']
    k = b['control_context']
    packet = r.get('IMU_prediction') or {}
    gravity = packet.get('gravity_body')
    tilt = math.atan2(math.hypot(*gravity[:2]), -gravity[2]) if vector(gravity, 3) else math.inf
    heading = h.get('last_goal_heading_rad') if exit else c.get('goal_heading')
    goal_s = h.get('last_goal_fix_s') if exit else c.get('goal_s')
    anchor = m.get('ball_capture_timestamp_s')
    age = m.get('ball_age_s')
    return dict(
        epoch=b.get('epoch') == k.get('epoch') == r.get('navigation_epoch'),
        capture=fresh(t, b.get('last_capture_s'), .2),
        guard=c.get('guard_safe') is True and fresh(t, c.get('guard_s'), .2),
        memory=m.get('memory_available') is True,
        anchor_age=number(age) and 0 <= age <= 6.5 and number(anchor) and eq(t-anchor, age),
        original_anchor_matches=eq(trace.get('anchor_capture_s'), anchor),
        enough_integrated_reads=m.get('samples_since_capture', 0) >= 8,
        memory_read_age=number(m.get('sensor_read_age_s')) and 0 <= m['sensor_read_age_s'] <= .08,
        body_sensor=packet.get('orientation_valid') is True and fresh(t, packet.get('read_timestamp_s'), .08) and tilt <= .35,
        point=domain(m.get('point_level_m')) and eq(trace.get('point_level_m'), m.get('point_level_m')),
        heading=number(heading) and abs(heading) <= math.pi/12 and eq(trace.get('heading_rad'), heading),
        goal=fresh(t, goal_s, 2.),
        commanded_stopped=zero(k.get('actual_velocity'), 3),
    )


def audit(pp, cc, manifest, check):
    """Returns verified broad-domain kick timestamps and inspectable decisions."""
    enabled = (manifest.get('blind_arrival') or {}).get('profile') == 'near-blind-v1'
    details = dict(enabled=enabled, entries=[], exits=[], withdrawals=[])
    if not enabled:
        check('blind_disabled_has_no_evidence', not any(r.get('blind_arrival') for r in pp+cc))
        return set(), details
    # Executor advances physics, stamps captures at tick+1, then runs the next
    # policy tick at that same timestamp. Bind the actual recorded capture used
    # by the policy, including this legitimate equality, never a future image.
    capture_times = [r['timestamp_s'] for r in cc]
    history = []
    previous_epoch = None
    last_capture = None
    visible = None
    missing = 0
    for c in cc:
        t = c['timestamp_s']
        b = c.get('blind_arrival') or {}
        context = b.get('capture_context') or {}
        # Control exceptions do not claim a new blind-gate capture; stale context
        # must not be counted as another missing image.
        if not eq(context.get('capture_s'), t):
            history.append(None)
            continue
        epoch = c.get('navigation_epoch')
        if epoch != previous_epoch:
            visible = None
            missing = 0
            last_capture = None
            previous_epoch = epoch
        if last_capture is not None and t-last_capture > .2+1e-9:
            missing = 0
        metric = c.get('public_metric_ball')
        metric_valid = bool(metric and eq(metric.get('timestamp_s'), t))
        check('blind_capture_metric_provenance', context.get('complete_metric_ball') == metric_valid, t,
              dict(context=context, public_metric_ball=metric))
        check('blind_capture_epoch_binding', context.get('epoch') == b.get('epoch') == epoch, t)
        if metric_valid:
            visible = t
            missing = 0
            check('blind_visible_metric_withdraws_latch', not b.get('latched'), t)
        else:
            missing += 1
        last_capture = t
        check('blind_missing_capture_count_recomputed', b.get('missing_capture_count') == missing, t)
        check('blind_visible_capture_time_recomputed', eq(b.get('last_visible_s'), visible), t)
        history.append(dict(epoch=epoch, visible=visible, missing=missing, capture=t, context=context))

    entries = {}
    exited = set()
    revoked_entries = set()
    previous = None
    seen_exit = set()
    for i, r in enumerate(pp):
        t = r['timestamp_s']
        b = r.get('blind_arrival')
        if not b:
            check('blind_enabled_policy_has_status', False, t)
            previous = r
            continue
        k = b.get('control_context') or {}
        h = r['public_handoff']
        entry = b.get('entry')
        key = None if not entry else (entry.get('epoch'), entry.get('t'))
        if previous and previous.get('navigation_epoch') == r.get('navigation_epoch'):
            check('blind_actual_previous_command_binding', eq(k.get('actual_velocity'), previous['actual_velocity_command']) and eq(k.get('actual_head'), previous['actual_head_command_rad']), t)
        if entry and eq(entry.get('t'), t) and key not in entries:
            index = bisect.bisect_right(capture_times, t+1e-9)-1
            latest = next((history[j] for j in range(index, -1, -1)
                           if history[j] is not None and eq(history[j]['capture'], b.get('last_capture_s'))), None)
            checks = common(r, entry)
            checks.update(
                allowed=entry.get('allowed') is True,
                entry_owner=entry.get('handoff_state') in ('APPROACH', 'HEAD_RETURN') and entry.get('fine_state') in ('IDLE', 'SETTLE', 'REACQUIRE'),
                live_latch=b.get('latched') is True and b.get('attempted') is True,
                epoch_same=entry.get('epoch') == r.get('navigation_epoch'),
                authentic_visible_history=bool(latest and latest['epoch'] == r.get('navigation_epoch') and number(latest['visible']) and latest['visible'] < t),
                three_missing_captures=bool(latest and latest['missing'] >= 3 and latest['missing'] == entry.get('missing_capture_count')),
                history_time_binding=bool(latest and eq(latest['visible'], entry.get('last_metric_visible_s')) and eq(latest['capture'], b.get('last_capture_s'))),
                single_entry_epoch=not any(old[0] == key[0] for old in entries),
                ownership_transferred=h['state'] == 'HEAD_RETURN' and r['public_neutral_fine_step']['state'] == 'IDLE' and not r['public_neutral_fine_step']['override'],
                command_zero=zero(r['actual_velocity_command'], 3),
                no_truth=entry.get('truth_used') is False and entry.get('validated_kick_domain') is False,
            )
            for name, value in checks.items():
                check('blind_entry_'+name, value, t)
            entries[key] = dict(valid=all(checks.values()), record=entry)
            details['entries'].append(dict(t=t, epoch=key[0], checks=checks))
        if key and not b.get('latched') and (not previous or (previous.get('blind_arrival') or {}).get('latched')):
            revoked_entries.add(key)
            details['withdrawals'].append(dict(t=t, epoch=key[0], entry_s=key[1]))
        if key in revoked_entries:
            check('blind_revoked_latch_not_resurrected', not b.get('latched'), t)
        if key in entries and b.get('latched') and h['state'] == 'HEAD_RETURN':
            continuity = common(r, entry)
            # Arrival geometry and goal refresh are evaluated at the quiet exit;
            # original anchor and present sensor/floor validity hold throughout.
            for name, value in continuity.items():
                if name not in ('point', 'heading', 'goal'):
                    check('blind_preparation_'+name, value, t)
        ex = b.get('exit')
        if ex and eq(ex.get('t'), t) and (key, t) not in seen_exit:
            seen_exit.add((key, t))
            checks = common(r, ex, exit=True)
            checks.update(
                valid_entry=key in entries and entries[key]['valid'],
                same_epoch=key is not None and ex.get('epoch') == key[0] == r.get('navigation_epoch'),
                original_anchor=key in entries and eq(ex.get('anchor_capture_s'), entries[key]['record'].get('anchor_capture_s')),
                original_entry_time=key is not None and eq(ex.get('entry_s'), key[1]),
                actual_head_zero=zero(r['actual_head_command_rad'], 4) and zero(k.get('actual_head'), 4),
                actual_body_zero=zero(r['actual_velocity_command'], 3),
                original_memory_age=eq(ex.get('anchor_age_s'), h['memory'].get('ball_age_s')),
                real_goal_time=eq(ex.get('goal_s'), h.get('last_goal_fix_s')),
                real_guard_time=eq(ex.get('guard_s'), b['capture_context'].get('guard_s')) and ex.get('guard_safe') == b['capture_context'].get('guard_safe'),
            )
            if ex.get('allowed'):
                for name, value in checks.items():
                    check('blind_exit_'+name, value, t)
                check('blind_exit_live_latch', b.get('latched') and key not in revoked_entries, t)
                if all(checks.values()) and b.get('latched') and key not in revoked_entries:
                    exited.add(t)
            else:
                check('blind_failed_exit_revokes_latch', not b.get('latched'), t)
            details['exits'].append(dict(t=t, allowed=ex.get('allowed'), checks=checks))
        if r.get('kick_requested') and (not previous or not previous.get('kick_requested')) and b.get('latched'):
            check('blind_kick_requires_this_tick_valid_exit', t in exited, t)
        previous = r
    return exited, details
