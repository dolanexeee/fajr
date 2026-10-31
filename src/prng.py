"""Deterministic PRNG (SplitMix64) — the ONLY randomness source in the sim.

Its whole state is two JSON-friendly values (an int and an optional float),
so a checkpoint can capture it exactly and a resumed run reproduces the
uninterrupted run bit-for-bit. The stdlib `random` module is never used
inside the simulation, and neither is wall-clock time.
"""
import math

_MASK = (1 << 64) - 1


class PRNG:
    def __init__(self, seed: int):
        self.state = seed & _MASK
        self._gauss_spare = None

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        return z ^ (z >> 31)

    def uniform(self) -> float:
        """Uniform in [0, 1)."""
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def range(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.uniform()

    def randint(self, lo: int, hi: int) -> int:
        """Inclusive both ends."""
        return lo + int(self.uniform() * (hi - lo + 1))

    def chance(self, p: float) -> bool:
        return self.uniform() < p

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        if self._gauss_spare is not None:
            v = self._gauss_spare
            self._gauss_spare = None
            return mu + sigma * v
        # Box–Muller
        u1 = max(self.uniform(), 1e-12)
        u2 = self.uniform()
        r = math.sqrt(-2.0 * math.log(u1))
        t = 2.0 * math.pi * u2
        self._gauss_spare = r * math.sin(t)
        return mu + sigma * (r * math.cos(t))

    def to_dict(self):
        return {"state": self.state, "gauss_spare": self._gauss_spare}

    @classmethod
    def from_dict(cls, d):
        p = cls(0)
        p.state = int(d["state"]) & _MASK
        p._gauss_spare = d["gauss_spare"]
        return p
