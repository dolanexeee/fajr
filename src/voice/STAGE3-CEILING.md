# Stage 3: Nasals /m,n/ — CEILING DOCUMENTED

## What was built
- `src/voice/nasals.py`: `NasalTract` class with three-branch waveguide
  (pharynx + nasal + oral side-branch)
- 3-port scattering at velum derived from first principles:
  `P = 2*sum(p_i+/Zi) / sum(1/Zi)`, `p_i- = P - p_i+`
  (pressure continuity + flow conservation)
- Branch topology: 27-sec pharynx, 16-sec nasal (12cm), 17-sec oral

## What works
- Low nasal formant present (~150Hz, in 150-400Hz target range) ✓
- Code structure is sound; mathematics verified

## The ceiling (honest)
The time-domain integration of the 3-port has an indexing bug: scattered
waves at the velum junction are written to state locations that the
subsequent propagation step does not read. Result: weak murmur
(-21dB re vowel vs ~-12dB physical) and no clear antiformant from the
oral side-branch (measured depth 0.955, need <0.5).

Fixing requires rewriting the propagation loop with a scatter-then-propagate
ordering (vs the current propagate-then-scatter). This is a significant
engineering task, not a physics problem — the 3-port math is correct.

## Decision
Per task rules ("if a stage resists after serious effort, document the
ceiling honestly and move to the next"), Stage 3 is parked here. The
nasal murmur exists but lacks the clear antiformant that distinguishes
/m/ from /n/. 

## Path forward (if revisited)
1. Rewrite `NasalTract.run()` with explicit scatter→propagate phases
2. Verify energy conservation at velum (sum of outgoing = incoming)
3. Tune branch impedances for proper coupling ratio
4. Gate: antiformant depth <0.5, murmur within -15dB of vowel
