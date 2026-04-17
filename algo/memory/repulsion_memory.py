from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from algo.planning.grid_map import GridMap2D
from algo.planning.repulsion import add_path_repulsion


@dataclass(frozen=True)
class RepulsionMemoryConfig:
    """External repulsion memory config (per initial)."""

    decay: float = 0.99
    max_value: float = 50.0
    radius_cells: int = 2
    strength_success: float = 2.0
    strength_failure: float = 0.0

    def strength(self, success: bool) -> float:
        return float(self.strength_success if success else self.strength_failure)


def _traj_xy_to_path_rc(
    grid: GridMap2D,
    traj_xy: Sequence[Tuple[float, float]],
) -> List[Tuple[int, int]]:
    path_rc: List[Tuple[int, int]] = []
    last: Optional[Tuple[int, int]] = None
    for x, y in traj_xy:
        r, c = grid.world_to_grid(float(x), float(y))
        if not grid.in_bounds(r, c):
            continue
        rc = (int(r), int(c))
        if last is not None and rc == last:
            continue
        path_rc.append(rc)
        last = rc
    return path_rc


class PerInitialRepulsionMemory:
    """A simple per-initial external memory bank on a 2D grid.

    Each initial_id maps to a (H,W) float32 array. The memory is updated by:
      memory *= decay
      memory += add_path_repulsion(path_rc, strength, radius)
      clip to [0, max_value]
    """

    def __init__(
        self,
        grid: GridMap2D,
        initial_ids: Iterable[int],
        config: RepulsionMemoryConfig = RepulsionMemoryConfig(),
        seed: int = 1,
    ) -> None:
        self.grid = grid
        self.config = config
        self.rng = np.random.default_rng(int(seed))

        self._mem: Dict[int, np.ndarray] = {}
        for iid in initial_ids:
            self._mem[int(iid)] = np.zeros_like(self.grid.occupancy, dtype=np.float32)

    def get(self, initial_id: int) -> np.ndarray:
        return self._mem[int(initial_id)]

    def reset(self, initial_id: int) -> None:
        self._mem[int(initial_id)].fill(0.0)

    def reset_all(self) -> None:
        for m in self._mem.values():
            m.fill(0.0)

    def update_with_trajectory(
        self,
        initial_id: int,
        traj_xy: Sequence[Tuple[float, float]],
        success: bool,
    ) -> None:
        m = self._mem[int(initial_id)]
        m *= float(self.config.decay)

        strength = float(self.config.strength(success))
        if strength <= 0.0:
            return

        path_rc = _traj_xy_to_path_rc(self.grid, traj_xy)
        if len(path_rc) == 0:
            return

        add_path_repulsion(
            penalty_map=m,
            path_rc=path_rc,
            strength=strength,
            radius_cells=int(self.config.radius_cells),
        )
        if float(self.config.max_value) > 0.0:
            np.clip(m, 0.0, float(self.config.max_value), out=m)

