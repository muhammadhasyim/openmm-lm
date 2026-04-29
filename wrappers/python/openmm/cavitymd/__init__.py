"""C2F (Cavity Configurational Feedback) protocol for OpenMM cavity-MD.

Provides coupling modulation, fictive temperature computation, feedback
controllers, and a simulation orchestrator for cavity molecular dynamics
with the C2F cooling protocol.
"""

from .constants import Units
from .variants import (
    CouplingVariant,
    ConstantVariant,
    StepVariant,
    SquareWaveVariant,
    AdaptiveSquareWaveVariant,
)
from .empirical import EmpiricalTemperatureData
from .trackers import ElapsedTimeTracker, EnergyTracker, TemperatureTracker
from .feedback import EmpiricalTemperatureFeedback, GradientDescentFeedback
from .thermostats import DualThermostat
from .simulation import (
    CavityMDSimulation,
    assign_force_groups,
    setup_gpu_square_wave,
    setup_gpu_step,
    setup_gpu_decaying_step,
    setup_gpu_adaptive_square_wave,
    setup_multimode_adaptive_square_wave,
)

__all__ = [
    "Units",
    "CouplingVariant",
    "ConstantVariant",
    "StepVariant",
    "SquareWaveVariant",
    "AdaptiveSquareWaveVariant",
    "EmpiricalTemperatureData",
    "ElapsedTimeTracker",
    "EnergyTracker",
    "TemperatureTracker",
    "EmpiricalTemperatureFeedback",
    "GradientDescentFeedback",
    "DualThermostat",
    "CavityMDSimulation",
    "assign_force_groups",
    "setup_gpu_square_wave",
    "setup_gpu_step",
    "setup_gpu_decaying_step",
    "setup_gpu_adaptive_square_wave",
    "setup_multimode_adaptive_square_wave",
]
