"""Adapt frozen scoring to the onboard physical sampling period; retain scene goal geometry."""
from pathlib import Path
import hashlib,sys,types

def install(dt, record):
    import contact_scoring, mechanical_scoring
    rows = []
    for original in (contact_scoring, mechanical_scoring):
        path = Path(original.__file__)
        text = path.read_text()
        old = "dt = 0.005 if self.last_t is None else"
        assert text.count(old) == 1
        text = text.replace(old, f"dt = {dt!r} if self.last_t is None else")
        if original.__name__ == "contact_scoring":
            assert text.count("if dt > 0.005001:") == 1
            text = text.replace("if dt > 0.005001:", f"if abs(dt - {dt!r}) > 1e-6:")
            text = text.replace("Record contact at200Hz physical substeps, not50Hz controls", "Record every actual candidate physics step")
        private = types.ModuleType(original.__name__)
        private.__file__ = str(path)
        sys.modules[original.__name__] = private
        exec(compile(text, str(path)+"<candidate-clock>", "exec"), private.__dict__)
        if original.__name__ == "contact_scoring":
            # Clock adaptation must retain the active scene's goal geometry.
            # Re-executing imports otherwise restores the unrelated default field.
            private.goal_crossing = original.goal_crossing
        rows.append(dict(path=str(path), original_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         candidate_sha256=hashlib.sha256(text.encode()).hexdigest(), physics_dt=dt))
    cls = sys.modules["mechanical_scoring"].PhysicalGoalScorerV2
    summary = cls.summary
    def annotated(self):
        value = summary(self)
        value.update(evaluation_physics_dt_s=dt, evaluation_physics_hz=1/dt,
            qualification_claimed=False, candidate_clock_changed=True,
            physics_semantics_id="onboard100; named mechanical limits reused")
        return value
    cls.summary = annotated
    record["scorer_clock"] = rows
