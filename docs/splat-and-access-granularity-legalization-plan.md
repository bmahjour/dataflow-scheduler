# Splat and Access Granularity Legalization Refactoring Plan

---

## 1. Path-Expansion Cleanup & Rebase
* **Action:** Revert commits `8235b312ef955bb3f177967f377d549329e7ef11` and `a8a47dd0c300320b1880397ae9e9a5d7170f0d67` to restore `path-expansion` to its prior state.
* **Context & Next Steps:** 
  * Avoid placing splat modes on intermediate adapted stages (e.g., `MNILU: DDR -> L1`).
  * Rebase onto `bmahjour/matmul-fifo-relay-and-path-exp`, where path expansion is refactored from stage-based to **per-flow expansion**.
  * Address the issue from [PR #198 discussion #r4349455036](https://github.com/torch-spyre/dataflow-scheduler/pull/198#discussion_r4349455036) directly in the per-flow planner:
    * Keep intermediate memory hops as scalar `[1, 1, 1, 1] -> [1, 1, 1, 1]` copies with `transfer_mode = Copy`.
    * Keep intermediate scratchpads sized to scalar elements (`memref<1x1x1x1xf16>`).
    * Attach `transfer_mode = Splat` (`[1, 1, 1, 1] -> [64]`) exclusively to the terminal synthetic hop loading into the compute FIFO (`L1LU -> SFU FIFO`).

---

## 2. Simplify / Eliminate `scalar-broadcast-legalization`
* **Action:** Strip out memory-side transfer widening and hardware access granularity checking (`mustLegalizeLoad`) from `ScalarBroadcastLegalizationPass`.
* **Rationale & End State:**
  * Since front-end / KTIR lowering now produces explicit `transfer_mode = Splat` on loads into FIFOs, the reverse-engineering walk over `linalg.generic` indexing maps is no longer needed.
  * With `access-granularity-legalization` taking over all hardware burst/granularity widening, `scalar-broadcast-legalization` has no distinct responsibilities left.
  * **Retire/remove the pass completely**, folding any residual compute-side indexing map normalization or `read_from_fifo` type updates into `splat-legalization` or direct front-end lowering.

---

## 3. Simplify `splat-legalization`
* **Action:** Remove the `moveSplat()` logic and associated buffer/slot shrinking machinery from `SplatLegalizationPass`.
* **Rationale:** 
  * With per-flow path expansion placing splats directly on the terminal FIFO load, splats are already at their correct physical location.
  * `splat-legalization` returns to its core single responsibility:
    1. Validate load unit SIMD splat capabilities.
    2. Widen the load only if required by `L1LU` access granularity (e.g., `f32` requiring 2 elements).
    3. Annotate consuming `read_from_fifo` ops with `#ktdf.splat<first_subsimd_lane_to_each_subsimd>` to emit sub-SIMD broadcast shuffles.

---

## 4. Dedicated `access-granularity-legalization` Pass

### Purpose
Enforce hardware physical `access_granularity` constraints across **all** `ktdf.data_transfer` operations (memory-to-memory and memory-to-FIFO) independently of whether a broadcast/splat is involved (e.g., widening `MNILU` `[1, 1, 1, 1]` to `[1, 1, 1, 64]` for DDR bus constraints).

### Key Design Requirements
1. **Source Buffer Bounds Checking:**
   * When widening a load transfer (e.g., `MNILU: DDR -> L1` from `[1, 1, 1, 1]` to `[1, 1, 1, 64]`), the pass must inspect the source `memref` shape/allocation bounds.
   * Verify that the innermost dimension (or total contiguous allocation) is large enough to support the widened physical read (e.g., a slice over `memref<12x1x64x64xf16, "DDR">`), or diagnose/reject unpadded small buffers (e.g., `memref<1x1x1x1xf16, "DDR">`) that would cause out-of-bounds reads in physical memory without hardware masking.
2. **Loads vs. Stores:**
   * Differentiate read widening (which can safely read extra elements into padded intermediate buffers) from write widening (which requires hardware byte-enables or masking to avoid clobbering adjacent memory).
3. **Integration with Padding Modes:**
   * Leverage PR #198's `TransferMode::ZeroPadFront` / `ZeroPadBack` for sub-granule non-splat slices (boundary/tail tiles).

### Suggested Pipeline Placement

```text
...
fold-fifo-relay-stages
path-expansion
indirect-addr-buf-fill-legalization
-------------------------------------------------------------------------
[NEW] access-granularity-legalization   <-- (1) Legalizes physical transfer sizes, checks bounds & resizes L1 allocations
      splat-legalization                <-- (2) Validates & finalizes FIFO splats / sub-SIMD shuffles
-------------------------------------------------------------------------
normalize-scf-for-loops
canonicalize
tile-scf-for-loops
stage-coarsening
reduction-dim-chunking
...
```

### Why This Placement?
1. **Immediately Post-Path-Expansion:** Runs once all physical execution units (`MNILU`, `L1LU`, `L1SU`, `MNISU`) and intermediate memory tiers (DDR, L1) are fully instantiated and concrete.
2. **Before Loop Tiling & Coarsening:** Downstream transformations (loop tiling, double buffering, memory footprint analysis, and cost modeling) will observe the true physical memory allocations and hardware transfer sizes.
3. **Decoupled from `splat-legalization`:** Memory-side burst/granularity rules run first; SIMD/FIFO broadcast rules run second.
