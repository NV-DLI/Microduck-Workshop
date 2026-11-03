"""Install the onboard transport and geometry implementations once per Kit process."""
from pathlib import Path
import os
import sys

HERE = Path(__file__).resolve().parent
_installed = False
_affinity = None


def install():
    global _installed
    if _installed:
        return
    import detector_client
    import perception
    from detector_raw_client import make_session_class
    from floor_fast import fit_virtual_floor
    from scoring_clock import install as install_clock
    detector_client.DetectorSession = make_session_class(detector_client, server=HERE / "serve_detector.py")
    perception.fit_virtual_floor = fit_virtual_floor
    install_clock(.01, {})
    import sensor_panel
    import display_layout
    def attach(env, argv):
        if not sensor_panel.kit_window_requested(argv):
            return None
        # Each attachment owns its recorder and widgets. No previous round is reused.
        output = Path(argv[argv.index("--output") + 1])
        recorder = display_layout.Recorder(output.with_name(output.name + "-layout.json"),
            dict(wrapper_sha256={}, changed="display only", targets=dict(right=.12, bottom=.07)))
        cls = display_layout.make_panel_class(sensor_panel)
        attachment = sensor_panel.Attachment(env.beta_rig,
            panel_factory=lambda source: cls(source, recorder=recorder), role="onboard")
        return attachment
    sensor_panel.attach = attach
    _installed = True


def pin_main_thread():
    """Leave Kit workers untouched; select the faster cores on heterogeneous CPUs."""
    global _affinity
    if _affinity is not None or not hasattr(os, "sched_getaffinity"):
        return
    allowed = os.sched_getaffinity(0)
    maxima = {}
    for cpu in allowed:
        file = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq")
        if file.exists():
            maxima[cpu] = int(file.read_text())
    selected = {cpu for cpu, hz in maxima.items() if hz == max(maxima.values())} if maxima else set()
    _affinity = allowed
    if selected and selected < allowed:
        os.sched_setaffinity(0, selected)
