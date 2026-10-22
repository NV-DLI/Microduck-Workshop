"""Optional new-process estimator wrapper; no shared source edits or installation."""
import functools


def wrap_estimator(original_estimator, *, border_checker=None):
    """Preserve every old rejection; add border veto before metric consumers."""
    if getattr(original_estimator, "_observed_border_wrapped", False):
        raise ValueError("Border guard must be installed exactly once")
    if border_checker is None:
        from guard import check_observed_ball_border
        border_checker = check_observed_ball_border

    @functools.wraps(original_estimator)
    def estimate(rgb, box, **kwargs):
        previous = original_estimator(rgb, box, **kwargs)
        if previous.get("accepted") is not True:
            return previous
        check = border_checker(rgb, box)
        out = {**previous, "observed_image_border_guard": check}
        if not check["accepted"]:
            # Original timing/geometry diagnostics remain; no usable metric is
            # emitted or recorded as a fresh complete-ball view.
            out.update(accepted=False, reason=check["reason"], point_reference_m=None)
        return out

    estimate._observed_border_wrapped = True
    return estimate
