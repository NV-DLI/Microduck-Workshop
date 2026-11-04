"""Restart observation supervision without jumping the actual head command."""


def restart_head_feedback(previous):
    feedback = type(previous)()
    # Keep the actual sent command, bounded correction, clock and read sequence.
    # Only the failed observation/withdrawal attempt starts over. The existing
    # core withdraws its correction through the normal slew limit on the next tick.
    feedback.core = previous.core
    feedback.compensated = bool((abs(feedback.core.bias) > 1e-12).any())
    return feedback
