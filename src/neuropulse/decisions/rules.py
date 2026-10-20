"""Transparent research demonstration rules; no calibrated failure probabilities."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RuleState:
    last_cycle: int | None = None
    above: int = 0
    below: int = 0
    sustained: bool = False
    recommendation: str | None = None


@dataclass(frozen=True)
class DemoRules:
    critical: float = 20.0
    maintenance: float = 50.0
    combined: float = 80.0
    persistence: int = 3
    version: str = "demo-v1"

    def __post_init__(self) -> None:
        if not 0 <= self.critical <= self.maintenance <= self.combined or self.persistence <= 0:
            raise ValueError("Invalid rule thresholds/persistence")

    def apply(
        self, rul: float, cycle: int, anomaly: bool | None, prior: RuleState | None = None
    ) -> tuple[str, list[str], RuleState]:
        import math

        if not math.isfinite(rul) or rul < 0 or cycle <= 0:
            raise ValueError("Rules require finite nonnegative RUL and positive cycle")
        prior = prior or RuleState()
        state = prior if prior.last_cycle is None or cycle == prior.last_cycle + 1 else RuleState()
        above = state.above + 1 if anomaly is True else 0
        below = state.below + 1 if anomaly is False else 0
        sustained = state.sustained
        if above >= self.persistence:
            sustained = True
        if below >= self.persistence or anomaly is None:
            sustained = False
        if rul <= self.critical:
            recommendation, reasons = "critical_warning", [f"RUL_LE_{self.critical:g}"]
        elif rul <= self.maintenance:
            recommendation, reasons = "maintenance_recommended", [f"RUL_LE_{self.maintenance:g}"]
        elif sustained and rul <= self.combined:
            recommendation, reasons = (
                "maintenance_recommended",
                ["ANOMALY_SUSTAINED", f"RUL_LE_{self.combined:g}"],
            )
        elif sustained:
            recommendation, reasons = "monitor", ["ANOMALY_SUSTAINED"]
        else:
            recommendation, reasons = "normal", []
        return (
            recommendation,
            reasons,
            replace(
                state,
                last_cycle=cycle,
                above=above,
                below=below,
                sustained=sustained,
                recommendation=recommendation,
            ),
        )
