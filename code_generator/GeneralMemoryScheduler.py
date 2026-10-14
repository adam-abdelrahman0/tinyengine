# ----------------------------------------------------------------------
# Project: TinyEngine
# Title:   GeneralMemoryScheduler.py
#
# Reference papers:
#  - MCUNet: Tiny Deep Learning on IoT Device, NeurIPS 2020
#  - MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NeurIPS 2021
#  - MCUNetV3: On-Device Training Under 256KB Memory, NeurIPS 2022
# Contact authors:
#  - Wei-Ming Chen, wmchen@mit.edu
#  - Wei-Chen Wang, wweichen@mit.edu
#  - Ji Lin, jilin@mit.edu
#  - Ligeng Zhu, ligeng@mit.edu
#  - Song Han, songhan@mit.edu
#
# Target ISA:  ARMv7E-M
# ----------------------------------------------------------------------

import math

from .allocator.firstFit import FirstFit
from .constant import (
    FUSE_SGD_UPDATE_STR,
    FUSHION_CONFIG,
    INFERECE_WEIGHT_SIZE,
    TTYPE_INFERNECE,
    TTYPE_STATIC_WEIGHT,
    TTYPE_TRAINING_ACTIVATION,
    TTYPE_TRAINING_GRADIENT,
    TTYPE_TRAINING_WEIGHTS,
)


# Patch-based inference (InputResizer.py) renames the graph_idx of the tensor
# crossing the patch/normal-inference boundary by appending this suffix, to
# decouple its lifetime tracking from the patch loop's own hand-written C
# code (which this scheduler can't see into -- there's no real op "producing"
# that tensor for it to anchor a lifetime on). The renamed tensor and its
# original-numbered self are still the same logical value, so any code
# comparing graph_idx strings for tensor identity must strip this suffix
# first, or a downstream consumer referencing the original (un-suffixed)
# graph_idx will silently fail to match.
PATCH_BOUNDARY_SUFFIX = "_start_normal_infernece_block"


def _base_graph_idx(idx):
    idx = str(idx)
    return idx[: -len(PATCH_BOUNDARY_SUFFIX)] if idx.endswith(PATCH_BOUNDARY_SUFFIX) else idx


class GeneralMemoryScheduler:
    def __init__(
        self,
        layer,
        tflite_op=False,
        dummy_address=False,
        memory_limit=10 * 1024 * 1024,
        inplace=True,
        outputTables=None,
        mem_visual_path="codegen/allocation.png",
        VisaulizeTrainable=True,
        sort_by_lifetime=False,
        sort_by_start=False,
        fp_requantize=False,
    ):
        self.layer = layer
        self.fp_requantize = fp_requantize
        self.heads = 0
        self.buffers = {
            "input_output": 0,
            "residual": 0,
            "im2col": 0,
            "kernel": 0,
            "feature": 0,
            "trainable": 0,
        }  # for feature pyramid
        # overall memory info
        self.peakmem = 0
        self.flash = 0
        self.bias = 0
        self.scale = 0
        self.code = 0
        self.allocator = FirstFit(memory_limit, sort_by_lifetime, sort_by_start)
        self.outputTables = outputTables
        self.USE_INPLACE = inplace
        # off by default on purpose: this is a newer, separately-opted-into
        # optimization (see _reclaimPointwiseGrowthSpace) layered on top of
        # USE_INPLACE's existing (always-on) aliasing. Existing callers that
        # never touch this flag get byte-identical scheduling to before it
        # existed -- only tflite_convert.py's --pointwise-overwrite flag
        # turns it on.
        self.ENABLE_POINTWISE_OVERWRITE = False
        self.mem_visual_path = mem_visual_path
        self.tflite_op = tflite_op
        self.dummy_address = dummy_address
        self.VisaulizeTrainable = VisaulizeTrainable

        # for showing layer-wise memory usage
        self.layermem = []

    def _isTranable(self, name):
        for o in self.outputTables:
            if isinstance(name, str) and o.name in name:
                return True
        return False

    def _hasOtherConsumer(self, graph_idx, excluding_op):
        """True if some op other than `excluding_op` also reads a tensor with
        this graph_idx -- e.g. a residual/skip ADD further downstream that
        needs this exact value, not just the op immediately consuming it.
        The in-place DEPTHWISE_CONV_2D aliasing below computes its output by
        overwriting its input's own buffer (a real in-place C kernel call,
        not just an address-sharing bookkeeping trick) -- safe only when
        nothing else still needs that original value. Confirmed via a real
        on-device/QEMU logit mismatch tracing back to exactly this: a
        depthwise conv's in-place output silently destroyed a tensor a
        later residual ADD still needed, with no error at codegen or build
        time since the scheduler never checked for this second consumer.
        Patch-based inference (InputResizer.py) renames the graph_idx of a
        tensor crossing the patch/normal-inference boundary by appending
        "_start_normal_infernece_block" (to decouple its lifetime tracking
        from the patch loop's own hand-written C code, which the scheduler
        can't see into). That rename must not defeat this check: the renamed
        tensor and its original-numbered self are still the same logical
        value, and a downstream op may still reference it by the original,
        un-suffixed graph_idx. Strip the suffix from both sides before
        comparing so the two spellings match.
        """
        target = _base_graph_idx(graph_idx)
        for other in self.layer:
            if other is excluding_op:
                continue
            for inp in other.input_tensors:
                if _base_graph_idx(inp.graph_idx) == target:
                    return True
        return False

    def _hasLaterConsumer(self, graph_idx, from_idx):
        """True if some op AFTER position from_idx in self.layer still reads
        a tensor with this graph_idx. Unlike _hasOtherConsumer (which looks
        at every op, regardless of position, and is for deciding whether a
        PRODUCER may destroy its OWN input in place as it writes its output
        -- any consumer anywhere else is a problem there), this is for
        deciding whether it's safe to recycle a tensor's buffer for a NEW
        value once every op that still needed the OLD value has already run.
        self.layer is in program order, so an "other consumer" positioned
        before from_idx already read the value and is irrelevant; only a
        later one still matters."""
        target = _base_graph_idx(graph_idx)
        for other in self.layer[from_idx + 1 :]:
            for inp in other.input_tensors:
                if _base_graph_idx(inp.graph_idx) == target:
                    return True
        return False

    def _findRectangle(self, allocator_idx):
        for r in self.allocator.rectangles:
            if r["idx"] == allocator_idx:
                return r
        return None

    def _effectiveSize(self, r, asker_start):
        """Mirrors FirstFit.fit()'s own stride2_inplace_idx shrink rule
        (see firstFit.py): once the op asking is past stride2_inplace_idx,
        a stride-2 depthwise conv has already run in-place on this
        rectangle and downsampled it, so only a quarter of its nominal
        size is still actually occupied -- the other three quarters were
        freed by that same in-place write. All three collision-checking
        helpers below (_regionFreeDuring, _blockingRectangles, and
        _tightestFit's candidate generation) need this, or they'd treat
        an already-shrunk rectangle as if it were still full-size and
        wrongly refuse a placement fit() itself would already allow.
        """
        if r["stride2_inplace_idx"] is not None and asker_start > r["stride2_inplace_idx"]:
            return math.ceil(r["size"] / 4)
        return r["size"]

    def _regionFreeDuring(self, addr_start, addr_size, t_start, t_end, exclude_idx):
        """True if no OTHER placed rectangle's address range overlaps
        [addr_start, addr_start+addr_size) during a time window that
        overlaps [t_start, t_end)."""
        addr_end = addr_start + addr_size
        for r in self.allocator.rectangles:
            if r["idx"] in exclude_idx or r["placement"] == -1:
                continue
            if r["start"] < t_end and t_start < r["end"]:  # time overlap
                r_end = r["placement"] + self._effectiveSize(r, t_start)
                if r["placement"] < addr_end and addr_start < r_end:  # address overlap
                    return False
        return True

    def _reclaimPointwiseGrowthSpace(self):
        """A stride-1, int8, 1x1 CONV_2D (a pointwise channel-expansion,
        e.g. StaticSeparableTransition's pw) whose input has no other
        consumer can write its output directly over its own input's
        address range plus whatever free space sits immediately below it,
        instead of needing a brand-new, separately-sized buffer.

        Safe because convolve_1x1_s8*.c's kernels already copy each
        pixel's full input into a small separate im2col scratch buffer
        BEFORE computing and writing that same pixel's output (verified by
        reading the kernel source, not assumed) -- so the write for pixel p
        can never clobber pixel p's own not-yet-read input, only an
        EARLIER pixel's input, which has already been consumed. Output
        grows (output_c - input_c) bytes/pixel faster than input shrinks
        away, so the write-front catches the read-front only after
        (num_pixels - 1) * growth bytes -- as long as the free gap below
        the input is at least that large, the sweep finishes before it
        catches up. Confirmed empirically against this exact layer shape
        before implementing: forcing naive graph_idx aliasing (no gap
        accounting) made peak WORSE, not better -- this gap check is what
        makes it actually safe instead of just hopeful.
        """
        self._merge_group = {}
        for i, op in enumerate(self.layer):
            p = op.params
            if not (
                p.get("op") == "CONV_2D"
                and p.get("input_dtype") == "int8"
                and not self.tflite_op
                and p.get("kernel_h") == 1
                and p.get("kernel_w") == 1
                and p.get("stride_h") == 1
                and p.get("stride_w") == 1
            ):
                continue
            if op.get_layer_info().get("is_patch"):
                continue  # patch-internal addressing is its own scheme -- not accounted for here
            if self._hasLaterConsumer(op.input_tensors[0].graph_idx, i):
                continue  # input still needed later -- can't destroy it
            growth = p["output_c"] - p["input_c"]
            if growth < 0:
                continue  # output is SMALLER per-pixel -- the separate strict output_c<input_c
                          # aliasing rule elsewhere handles that case; nothing to do here
            # growth == 0 (same channel count) gives gap_needed == 0 below: the tightest,
            # safest case of all -- a pure zero-offset alias, identical in spirit to the
            # existing DEPTHWISE_CONV_2D in-place rule, just for a 1x1 conv. (A same-size
            # aliasing rule WAS tried earlier and caused a regression -- but that was a
            # *different*, zero-gap-UNAWARE implementation that just set matching graph_idx
            # directly, with no lifetime/collision accounting; _tryApplyGapOverlap's
            # explicit gap_needed=0 case goes through the exact same collision-checked,
            # peak-verified, rollback-safe path as every strictly-positive growth case here.)
            num_pixels = p["output_h"] * p["output_w"]
            gap_needed = (num_pixels - 1) * growth
            in_rec = self._findRectangle(op.input_tensors[0].allocator_idx)
            out_rec = self._findRectangle(op.output_tensors[0].allocator_idx)
            if in_rec is None or out_rec is None or in_rec["placement"] == -1:
                continue
            self._tryApplyGapOverlap(in_rec, out_rec, gap_needed)

    def _stemConvHasViolation(self, H_in, W_in, C_in, C_out, stride, pad, kernel, gap):
        """Simulates convolve_s8_kernel3_inputch3_stride2_pad1.c's exact
        access pattern at the byte level: row-major output pixels, each
        reading its full kernel x kernel x C_in window via im2col into a
        separate scratch buffer BEFORE computing and writing that same
        pixel's C_out output values (verified by reading the kernel
        source) -- the same read-before-write property the 1x1 case
        relies on, just across a 2D window instead of a single column.
        Checks whether output_base = input_base - gap ever lets a write
        reach an input byte some later-processed output pixel still needs
        to read. Returns on the FIRST violation (only used as a yes/no
        check by the binary search below, never to enumerate violations).

        Unlike the 1x1 case, adjacent output ROWS share input rows (stride
        2 with a 3-tall kernel re-reads the boundary row), so there's no
        clean closed-form gap formula the way (num_pixels-1)*growth was
        for 1x1 -- this simulates the real access order directly instead
        of trusting a hand-derived expression for a 2D-window kernel.
        """
        H_out = (H_in + 2 * pad - kernel) // stride + 1
        W_out = (W_in + 2 * pad - kernel) // stride + 1
        in_size = H_in * W_in * C_in
        read_done = bytearray(in_size)
        for i_out_y in range(H_out):
            base_y = i_out_y * stride - pad
            for i_out_x in range(W_out):
                base_x = i_out_x * stride - pad
                for ky in range(kernel):
                    y = base_y + ky
                    if not (0 <= y < H_in):
                        continue
                    row_start = y * W_in
                    for kx in range(kernel):
                        x = base_x + kx
                        if 0 <= x < W_in:
                            start = (row_start + x) * C_in
                            for c in range(C_in):
                                read_done[start + c] = 1
                out_pix = i_out_y * W_out + i_out_x
                write_start = out_pix * C_out - gap
                for c in range(C_out):
                    idx = write_start + c
                    if 0 <= idx < in_size and not read_done[idx]:
                        return True
        return False

    def _minSafeGapStemConv(self, H_in, W_in, C_in, C_out, stride, pad, kernel):
        """Binary search for the smallest gap where _stemConvHasViolation
        is False. Monotonic and safe to binary-search: a bigger gap only
        ever moves every write address strictly lower while every read
        address is unaffected, so safety can only improve as gap grows --
        confirmed empirically across the real shape and a small toy shape
        before relying on it. gap = out_size is always a safe upper bound:
        it pushes every write address strictly negative (out of the
        input's own address range entirely), so it can never collide.
        """
        H_out = (H_in + 2 * pad - kernel) // stride + 1
        W_out = (W_in + 2 * pad - kernel) // stride + 1
        out_size = H_out * W_out * C_out
        lo, hi = 0, out_size
        while lo < hi:
            mid = (lo + hi) // 2
            if self._stemConvHasViolation(H_in, W_in, C_in, C_out, stride, pad, kernel, mid):
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _reclaimStemConvOverlap(self):
        """The network's stem conv (3x3, stride 2, pad 1, int8, exactly 3
        input channels -- convolve_s8_kernel3_inputch3_stride2_pad1.c's
        own hardcoded assumption, confirmed by reading the dispatch logic
        in operators/conv2d.py) can overlap its output with its raw input
        image the same way the 1x1 growth case does, using a gap found by
        simulation (_minSafeGapStemConv) instead of a closed-form formula,
        since adjacent output rows here share input rows in a way the 1x1
        case's per-pixel independence doesn't have.
        """
        for i, op in enumerate(self.layer):
            p = op.params
            if not (
                p.get("op") == "CONV_2D"
                and p.get("input_dtype") == "int8"
                and not self.tflite_op
                and p.get("kernel_h") == 3
                and p.get("kernel_w") == 3
                and p.get("stride_h") == 2
                and p.get("stride_w") == 2
                and p.get("padding") == 1
                and p.get("input_c") == 3
            ):
                continue
            if op.get_layer_info().get("is_patch"):
                continue
            if self._hasLaterConsumer(op.input_tensors[0].graph_idx, i):
                continue
            in_rec = self._findRectangle(op.input_tensors[0].allocator_idx)
            out_rec = self._findRectangle(op.output_tensors[0].allocator_idx)
            if in_rec is None or out_rec is None or in_rec["placement"] == -1:
                continue
            gap_needed = self._minSafeGapStemConv(
                p["input_h"], p["input_w"], p["input_c"], p["output_c"], p["stride_h"], p["padding"], p["kernel_h"]
            )
            self._tryApplyGapOverlap(in_rec, out_rec, gap_needed)

    def _tryApplyGapOverlap(self, in_rec, out_rec, gap_needed):
        """Enforces output_base <= input_base - gap_needed between in_rec
        and out_rec, mutating placements in place when a safe arrangement
        exists, else leaving the schedule untouched. Shared by every op
        family that reduces to this same relationship (currently the 1x1
        growth-conv case and the 3x3/stride-2 stem-conv case below) --
        only how each one computes gap_needed differs; once you have that
        number, making it true safely is identical.

        Both sides of the relationship are tried, because either address
        can be the one already pinned by the generic allocator:
          (a) input already placed -- slide output down to just touch it
              from below (e.g. stage1's pw reusing the freed input-image
              buffer below it).
          (b) output already sitting at the lowest address available
              (common: the generic allocator's largest-first sort put it
              at 0) -- nothing is below it to slide into, but input
              doesn't need to stay wherever it happened to land either: it
              only needs to sit at output_base + gap_needed, inside
              output's own span, instead of disjoint above it. Found by
              tracing a real case where (a) always lost (output already
              at address 0, so "cheaper than current" was never true)
              while input was sitting ~300KB higher than the gap actually
              requires.
        If in_rec is itself already a member of an earlier merge group
        (e.g. it's both this op's input AND some earlier op's merged
        output), neither (a) nor (b) alone can move it without breaking
        that earlier merge's own exact gap -- _tryGroupShiftAndMerge
        handles that by shifting the whole group rigidly instead. The same
        is true in the other direction: if out_rec is ALREADY a member of
        some earlier group (e.g. this op's own input chains into a
        different op family's output -- the 3x3 stem conv's output is
        also the 1x1 rule's own input), branch (a) must not move it
        directly either; it's simply skipped there (this specific op gets
        no overlap -- a bidirectional group-shift isn't attempted, judged
        not worth the added complexity for how rarely both sides are
        grouped at once).

        A mirror-image "use the slack ABOVE an inherited oversized slot
        instead" case (reverse pixel order) was drafted and deliberately
        removed -- NOT safe with convolve_1x1_s8*.c's existing forward-
        order kernels (verified both algebraically and with a concrete
        numeric example that it corrupts not-yet-read input). Every path
        here stays within the forward-order-safe relationship; only which
        side moves differs.
        """
        exclude_idx = {in_rec["idx"], out_rec["idx"]}
        out_rec_grouped = out_rec["idx"] in self._merge_group and len(self._merge_group[out_rec["idx"]]) > 1
        in_rec_grouped = in_rec["idx"] in self._merge_group and len(self._merge_group[in_rec["idx"]]) > 1
        applied = False
        new_out_start = in_rec["placement"] - gap_needed
        if not out_rec_grouped and 0 <= new_out_start < out_rec["placement"]:
            if self._regionFreeDuring(new_out_start, out_rec["size"], out_rec["start"], out_rec["end"], exclude_idx):
                out_rec["placement"] = new_out_start
                applied = True
            elif self._tryCascadeRelocate(out_rec, new_out_start, exclude_idx):
                applied = True
        if not applied and not in_rec_grouped and out_rec["placement"] != -1:
            new_in_start = out_rec["placement"] + gap_needed
            if new_in_start < in_rec["placement"]:
                if self._regionFreeDuring(new_in_start, in_rec["size"], in_rec["start"], in_rec["end"], exclude_idx):
                    in_rec["placement"] = new_in_start
                    applied = True
                elif self._tryCascadeRelocate(in_rec, new_in_start, exclude_idx):
                    applied = True
        if not applied:
            applied = self._tryGroupShiftAndMerge(in_rec, out_rec, gap_needed, exclude_idx)
        if applied:
            self._recordMergeGroup(in_rec["idx"], out_rec["idx"])
        return applied

    def _recordMergeGroup(self, idx_a, idx_b):
        """Tracks which rectangles have had their relative address offset
        pinned by a successful merge, transitively -- e.g. once 112 is
        merged into 114, and then 114 (as part of that same group) is
        later merged into 117, all three must move together as one rigid
        unit from then on, or an earlier merge's exact gap would break.
        """
        group = self._merge_group.get(idx_a, {idx_a}) | self._merge_group.get(idx_b, {idx_b})
        for idx in group:
            self._merge_group[idx] = group

    def _tryShiftGroupOutOfTheWay(self, group, shift, exclude_idx):
        """Rigidly shifts every member of `group` by `shift` -- preserving
        every internal gap relationship exactly, since every member moves
        by the identical delta -- then resolves whatever that newly
        collides with, using the same _tightestFit-based relocation
        _tryCascadeRelocate's own blocker loop uses.

        Refuses to touch any blocker that is itself a member of some
        OTHER already-established merge group (size > 1): moving just one
        member of a group this doesn't know about would silently break
        that other group's own gap relationship -- bounded to moving one
        group at a time rather than chasing that further.

        Returns True if every member landed collision-free (mutating
        self.allocator.rectangles in place); does NOT snapshot or roll
        back on failure -- every caller already has its own snapshot
        covering a broader change, so it owns that.
        """
        members = []
        for idx in group:
            m = self._findRectangle(idx)
            if m is None:
                return False
            members.append(m)

        for m in members:
            m["placement"] += shift

        full_exclude = set(group) | exclude_idx
        for m in members:
            if self._regionFreeDuring(m["placement"], m["size"], m["start"], m["end"], full_exclude):
                continue
            for blocker in self._blockingRectangles(m["placement"], m["size"], m["start"], m["end"], full_exclude):
                if blocker["idx"] in self._merge_group and len(self._merge_group[blocker["idx"]]) > 1:
                    return False
                old_bp = blocker["placement"]
                blocker["placement"] = -1
                candidate = self._tightestFit(blocker, full_exclude)
                if candidate is None or candidate == old_bp:
                    return False
                blocker["placement"] = candidate
        return True

    def _tryGroupShiftAndMerge(self, in_rec, out_rec, gap_needed, exclude_idx):
        """Branches (a)/(b) above fail when in_rec is pinned at the address
        floor by an EARLIER merge it's itself a member of (e.g. 114 is
        out_rec's own base address AND also has 112 nested inside it via
        its own merge) -- moving in_rec alone would break that earlier
        merge's exact gap relationship, but shifting the WHOLE group
        rigidly (via _tryShiftGroupOutOfTheWay) preserves every internal
        gap exactly, since they're all just translated together. Nothing
        is left modified unless the whole thing is both collision-free
        and peak-non-increasing.
        """
        group = self._merge_group.get(in_rec["idx"])
        if not group or len(group) < 2:
            return False
        shift = gap_needed - in_rec["placement"]
        if shift <= 0:
            return False  # in_rec already has enough room below it -- branch (b) would have used it

        old_peak = self._currentPeak()
        snapshot = {r["idx"]: r["placement"] for r in self.allocator.rectangles}

        ok = self._tryShiftGroupOutOfTheWay(group, shift, exclude_idx | {out_rec["idx"]})

        if ok:
            out_rec["placement"] = in_rec["placement"] - gap_needed
            if out_rec["placement"] < 0 or not self._regionFreeDuring(
                out_rec["placement"], out_rec["size"], out_rec["start"], out_rec["end"],
                set(group) | {out_rec["idx"]},
            ):
                ok = False

        # Strict "<", not "<=": a "doesn't make it worse right now" bar was
        # tried here, reasoning that a later op still unprocessed (e.g.
        # i=8's own merge) might be the thing this shift actually unblocks,
        # with its benefit not showing up in _currentPeak() until that
        # later op's own turn. Traced a real case where that bet didn't
        # pay off -- pinning the group higher to satisfy THIS merge (which
        # had no standalone benefit of its own) ended up costing far more
        # once later same-size merges needed the group to stay low, a cost
        # that outweighed anything it unblocked (402,054 vs 551,888 on the
        # real model, measured directly). A peak-neutral shift is kept only
        # when it's ALSO strictly better eventually; since that can't be
        # known in a single forward pass, require the improvement to be
        # visible immediately, same as every other accept-check here.
        if ok and self._currentPeak() < old_peak:
            return True

        for r in self.allocator.rectangles:
            r["placement"] = snapshot[r["idx"]]
        return False

    def _currentPeak(self):
        peak = 0
        for r in self.allocator.rectangles:
            if r["placement"] != -1:
                top = r["placement"] + r["size"]
                if top > peak:
                    peak = top
        return peak

    def _blockingRectangles(self, addr_start, addr_size, t_start, t_end, exclude_idx):
        """Rectangles whose address range collides with [addr_start,
        addr_start+addr_size) during a time window overlapping [t_start,
        t_end) -- the set _regionFreeDuring would have objected to."""
        addr_end = addr_start + addr_size
        blockers = []
        for r in self.allocator.rectangles:
            if r["idx"] in exclude_idx or r["placement"] == -1:
                continue
            if r["start"] < t_end and t_start < r["end"]:
                r_end = r["placement"] + self._effectiveSize(r, t_start)
                if r["placement"] < addr_end and addr_start < r_end:
                    blockers.append(r)
        return blockers

    def _tightestFit(self, rec, exclude_idx):
        """Correct lowest-valid-address search for rec, used instead of
        allocator.fit() during cascade relocation.

        The optimal placement for a rectangle is always either address 0
        or immediately above some other time-overlapping rectangle's top
        -- so checking exactly that finite candidate set via the
        already-verified _regionFreeDuring finds the true tightest slot.
        This matters because FirstFit.fit() mutates its own `slots` list
        (.remove()/.append()) while iterating over it with a plain `for s
        in slots:` loop -- a classic hazard that can skip a list item
        right after a removal shifts things down. Traced a real case where
        this caused fit() to settle for a valid-but-not-cheapest address
        (missing a free slot it should have found), rather than fix that
        shared routine (used by every model's baseline schedule and
        required to stay byte-identical), this search bypasses it only
        here, inside the opt-in cascade path.
        """
        candidates = {0}
        for r in self.allocator.rectangles:
            if r["idx"] in exclude_idx or r["placement"] == -1:
                continue
            if r["start"] < rec["end"] and rec["start"] < r["end"]:
                candidates.add(r["placement"] + self._effectiveSize(r, rec["start"]))
        for addr in sorted(candidates):
            if self._regionFreeDuring(addr, rec["size"], rec["start"], rec["end"], exclude_idx):
                return addr
        return None

    def _compactSchedule(self):
        """One-shot post-hoc tightening pass over the allocator's own
        output, before any pointwise-overwrite pairing is introduced.

        FirstFit.fit() (see _tightestFit's docstring for the exact
        mechanism) can leave an individual rectangle at a valid-but-not-
        cheapest address -- traced a real case of this independent of
        anything pointwise-overwrite-specific. Re-checks every already-
        placed rectangle with _tightestFit and slides it down whenever a
        strictly lower valid address exists, one rectangle at a time, each
        move touching nothing else -- so every move is non-increasing for
        the overall peak by construction, never needs a rollback. Repeats
        until a full pass makes no further change, since moving one
        rectangle down can free room for another.
        """
        for _ in range(len(self.allocator.rectangles) + 1):
            changed = False
            order = sorted(
                (r for r in self.allocator.rectangles if r["placement"] != -1),
                key=lambda r: r["placement"],
            )
            for r in order:
                old_placement = r["placement"]
                r["placement"] = -1
                candidate = self._tightestFit(r, {r["idx"]})
                if candidate is not None and candidate < old_placement:
                    r["placement"] = candidate
                    changed = True
                else:
                    r["placement"] = old_placement
            if not changed:
                break

    def _tryCascadeRelocate(self, out_rec, new_start, exclude_idx):
        """When _reclaimPointwiseGrowthSpace's direct region isn't free,
        try relocating whatever's blocking it elsewhere via _tightestFit
        (a direct, correct lowest-valid-address search -- see its
        docstring for why this is used instead of the allocator's own
        fit()), then re-check. All-or-nothing and peak-checked: if any
        blocker has nowhere better to go, or moving everything doesn't
        actually lower the overall peak, every change made here is rolled
        back and nothing is left modified -- this can only ever leave the
        schedule the same or smaller, never bigger.

        A blocker that is itself a member of an already-established merge
        group (size > 1, tracked in self._merge_group) is never relocated
        on its own -- moving just that one member, without its partner(s),
        would silently break that other merge's exact gap relationship
        (traced a real case of exactly this: it pushed a blocker to an
        address that violated the required output_base <= input_base -
        gap_needed relationship, an actual unsafe overlap, not just a
        missed optimization). Instead the WHOLE group is rigidly shifted
        out of the way via _tryShiftGroupOutOfTheWay, which preserves
        every internal gap exactly; if that isn't possible, the cascade
        is abandoned rather than touching the group partially.
        """
        blockers = self._blockingRectangles(new_start, out_rec["size"], out_rec["start"], out_rec["end"], exclude_idx)
        if not blockers:
            return False
        old_peak = self._currentPeak()
        snapshot = {r["idx"]: r["placement"] for r in self.allocator.rectangles}

        # fit() just finds *a* valid spot for a blocker -- left alone, it
        # often re-derives the exact same address it already had, since
        # nothing about ITS OWN constraints changed. A temporary reservation
        # over the region we actually want forces fit() to look elsewhere
        # instead of trivially handing back the status quo.
        reservation = {
            "start": out_rec["start"], "end": out_rec["end"], "size": out_rec["size"],
            "placement": new_start, "name": "_reservation", "idx": -2, "stride2_inplace_idx": None,
        }
        self.allocator.rectangles.append(reservation)

        moved_idx = set(exclude_idx)
        handled_groups = set()
        for blocker in blockers:
            group = self._merge_group.get(blocker["idx"])
            if group and len(group) > 1:
                gkey = frozenset(group)
                if gkey in handled_groups:
                    continue  # this blocker's whole group was already shifted for an earlier blocker in it
                shift = (new_start + out_rec["size"]) - blocker["placement"]
                if shift <= 0 or not self._tryShiftGroupOutOfTheWay(group, shift, exclude_idx | {out_rec["idx"]}):
                    self.allocator.rectangles.remove(reservation)
                    for r in self.allocator.rectangles:
                        r["placement"] = snapshot[r["idx"]]
                    return False
                handled_groups.add(gkey)
                moved_idx |= group
                continue
            old_placement = blocker["placement"]
            blocker["placement"] = -1  # treat this one as not-yet-placed during the search
            candidate = self._tightestFit(blocker, exclude_idx)
            if candidate is None or candidate == old_placement:
                # nowhere better than where it already was (or nowhere at
                # all) -- this cascade can't make progress, abandon it
                self.allocator.rectangles.remove(reservation)
                for r in self.allocator.rectangles:
                    r["placement"] = snapshot[r["idx"]]
                return False
            blocker["placement"] = candidate
            moved_idx.add(blocker["idx"])

        self.allocator.rectangles.remove(reservation)

        if not self._regionFreeDuring(new_start, out_rec["size"], out_rec["start"], out_rec["end"], moved_idx):
            # moving the named blockers wasn't enough -- something else
            # (not in our blocker list, e.g. revealed only after moving one
            # of them) still collides
            for r in self.allocator.rectangles:
                r["placement"] = snapshot[r["idx"]]
            return False

        out_rec["placement"] = new_start
        if self._currentPeak() >= old_peak:
            # technically-valid, but didn't actually help -- not worth the
            # extra moved buffers, roll back
            for r in self.allocator.rectangles:
                r["placement"] = snapshot[r["idx"]]
            return False
        return True

    def allocateMemory(self):
        # assign the same graph index for inplace operations
        # note: we need to handle stride == 2 for int8 depthwise to save memory
        if self.USE_INPLACE:
            for i, op in enumerate(self.layer):
                if (
                    op.params["op"] == "DEPTHWISE_CONV_2D"
                    and op.params["input_dtype"] == "int8"
                    and not self.tflite_op
                    and not self._hasOtherConsumer(op.input_tensors[0].graph_idx, op)
                ):
                    # set the idx of output and next layer input
                    previous_output_idx = op.output_tensors[0].graph_idx
                    op.output_tensors[0].graph_idx = op.input_tensors[0].graph_idx
                    if (
                        i + 1 < len(self.layer)
                        and len(self.layer[i + 1].input_tensors) > 0
                        and str(self.layer[i + 1].input_tensors[0].graph_idx) == str(previous_output_idx)
                    ):
                        self.layer[i + 1].input_tensors[0].graph_idx = op.input_tensors[0].graph_idx
                    # update following ops' tensors
                    for following_idx in range(i, len(self.layer)):
                        for cnt, inp_tensor in enumerate(self.layer[following_idx].input_tensors):
                            if str(inp_tensor.graph_idx) == str(previous_output_idx):
                                inp_tensor.graph_idx = op.input_tensors[0].graph_idx
                if (
                    self.ENABLE_POINTWISE_OVERWRITE
                    and op.params["op"] == "CONV_2D"
                    and op.params["input_dtype"] == "int8"
                    and not self.tflite_op
                    and op.params.get("kernel_h") == 1
                    and op.params.get("kernel_w") == 1
                    and op.params.get("stride_h") == 1
                    and op.params.get("stride_w") == 1
                    and op.params["output_c"] < op.params["input_c"]
                    and not self._hasLaterConsumer(op.input_tensors[0].graph_idx, i)
                ):
                    # Generalizes the DEPTHWISE_CONV_2D in-place case above to
                    # any stride-1 1x1 CONV_2D (full channel mixing, not just
                    # grouped) whose output has no more channels per pixel
                    # than its input -- e.g. a squeeze-style pointwise conv,
                    # or (once chained through a prior growth hop) a later
                    # hop that happens to fit inside an inherited oversized
                    # slot. Safe for the same reason convolve_1x1_s8*.c's
                    # kernels are already safe for DEPTHWISE_CONV_2D aliasing:
                    # they copy each pixel's full input into a small separate
                    # scratch buffer before computing and writing that same
                    # pixel's (no bigger) output, so the write for pixel p
                    # can never reach into pixel p+1's not-yet-read input --
                    # unlike the DEPTHWISE case, no stride/shape restriction
                    # beyond 1x1/stride-1 is needed, since output_c<=input_c
                    # alone guarantees the write-front never outruns the
                    # read-front. _hasLaterConsumer (not _hasOtherConsumer,
                    # see its own docstring) because a 1x1 conv's own
                    # immediate predecessor commonly shares its input's
                    # identity already (e.g. a DEPTHWISE_CONV_2D just aliased
                    # above) -- _hasOtherConsumer would see that EARLIER
                    # reference and wrongly refuse to alias.
                    previous_output_idx = op.output_tensors[0].graph_idx
                    op.output_tensors[0].graph_idx = op.input_tensors[0].graph_idx
                    for following_idx in range(i, len(self.layer)):
                        for cnt, inp_tensor in enumerate(self.layer[following_idx].input_tensors):
                            if str(inp_tensor.graph_idx) == str(previous_output_idx):
                                inp_tensor.graph_idx = op.input_tensors[0].graph_idx
                if (
                    op.params["op"] == "TRANSPOSE_CONV_2D"
                    and op.params["group"] == op.params["input_c"]
                    and op.params["group"] == op.params["output_c"]
                    and not self.tflite_op
                    and op.params["stride_h"] == 1
                    and op.params["stride_w"] == 1
                ):
                    # set the idx of output and next layer input
                    previous_output_idx = op.output_tensors[0].graph_idx
                    op.output_tensors[0].graph_idx = op.input_tensors[0].graph_idx
                    # update following ops' tensors
                    for following_idx in range(i, len(self.layer)):
                        for cnt, inp_tensor in enumerate(self.layer[following_idx].input_tensors):
                            if inp_tensor.graph_idx == previous_output_idx:
                                inp_tensor.graph_idx = op.input_tensors[0].graph_idx
                                # set the name in params which will be used later
                                if (
                                    cnt == 1
                                    and "CONV" in self.layer[following_idx].params["op"]
                                    and isinstance(self.layer[following_idx].params["weight_value"], str)
                                ):
                                    self.layer[following_idx].params["weight_value"] = op.input_tensors[0].graph_idx
                if (
                    op.params["op"] == "MUL"
                    and op.params["input_dtype"] == "int8"
                    and len(op.input_tensors) == 2
                    and str(op.input_tensors[0].graph_idx) == str(op.input_tensors[1].graph_idx)
                ):
                    # StarBlockV's self-gate act(f(x)) * x, when the graph
                    # optimizer has collapsed both operands to one shared
                    # tensor (see mul.py's generate_inference_str() -- the
                    # same input_idx == input2_idx check selects the
                    # mul_fpreq_inplace kernel over mul_fpreq for this case).
                    # Alias the output onto the input exactly like the
                    # DEPTHWISE_CONV_2D case above, so no separate SRAM
                    # buffer gets allocated for the multiply's output.
                    previous_output_idx = op.output_tensors[0].graph_idx
                    op.output_tensors[0].graph_idx = op.input_tensors[0].graph_idx
                    for following_idx in range(i, len(self.layer)):
                        for cnt, inp_tensor in enumerate(self.layer[following_idx].input_tensors):
                            if str(inp_tensor.graph_idx) == str(previous_output_idx):
                                inp_tensor.graph_idx = op.input_tensors[0].graph_idx
                if (
                    op.params["op"] == "ADD"
                    and op.params["input_dtype"] == "int8"
                    and not self.tflite_op
                    and len(op.input_tensors) == 2
                ):
                    # add_fpreq() (see its .c source) reads input1[k] and
                    # input2[k] into locals before writing output[k] -- a
                    # plain per-index loop with no cross-index dependency, so
                    # it's safe to write the result back into whichever
                    # operand's own buffer, as long as nothing still needs
                    # that operand's ORIGINAL value afterward. A residual
                    # add's skip operand was kept alive specifically so THIS
                    # op could read it (see _hasOtherConsumer above) -- once
                    # this op has read it, nothing else does, so its buffer
                    # is free to reuse for the sum instead of allocating a
                    # third, separate buffer. Check with _hasLaterConsumer
                    # (not _hasOtherConsumer): an operand's one other
                    # consumer is typically EARLIER in the program (e.g. the
                    # depthwise conv that read the skip value before this
                    # block's body ran), which already happened and doesn't
                    # block reuse -- only a consumer still to come would.
                    for candidate in (op.input_tensors[0], op.input_tensors[1]):
                        if not self._hasLaterConsumer(candidate.graph_idx, i):
                            previous_output_idx = op.output_tensors[0].graph_idx
                            op.output_tensors[0].graph_idx = candidate.graph_idx
                            for following_idx in range(i, len(self.layer)):
                                for cnt, inp_tensor in enumerate(self.layer[following_idx].input_tensors):
                                    if str(inp_tensor.graph_idx) == str(previous_output_idx):
                                        inp_tensor.graph_idx = candidate.graph_idx
                            break

        num_layers = len(self.layer)
        # add all trainable tensors as one tensor
        length_model = len(self.layer)
        trainable = 0
        weight_size = 0
        bias_size = 0
        for out_t in self.outputTables:
            if "bias" in out_t.name:
                dtype_multiplier = 4
                trainable += int(out_t.len * dtype_multiplier)
                bias_size += int(out_t.len * dtype_multiplier)
            elif "weight" in out_t.name:
                dtype_multiplier = INFERECE_WEIGHT_SIZE
                # find the conv2d owning the tensor
                conv_2d_op = None
                for lay in self.layer:
                    if "weight_name" in lay.params and out_t.name in lay.params["weight_name"]:
                        conv_2d_op = lay
                        break
                assert conv_2d_op is not None
                # check if it is partial
                if "first_k_channel" in conv_2d_op.params and conv_2d_op.params["first_k_channel"] is not None:
                    trainable += int(
                        out_t.len
                        * dtype_multiplier
                        * conv_2d_op.params["first_k_channel"]
                        / conv_2d_op.params["input_c"]
                    )
                    weight_size += int(
                        out_t.len
                        * dtype_multiplier
                        * conv_2d_op.params["first_k_channel"]
                        / conv_2d_op.params["input_c"]
                    )
                else:
                    trainable += int(out_t.len * dtype_multiplier)
                    weight_size += int(out_t.len * dtype_multiplier)
            else:
                pass
        if self.VisaulizeTrainable:
            self.allocator.addTensor(0, length_model, trainable, type=TTYPE_STATIC_WEIGHT)

        all_t_size = 0

        # Patch-based inference (InputResizer.py) tags the first op of the
        # normal-inference block with is_start_of_normal_inference_block and
        # forces ITS OWN input tensor's lifetime to start at 0 below, since
        # that tensor is synthesized by hand-written reassembly C code with
        # no real scheduler-tracked producer op. But that same synthesized
        # value can have OTHER consumers too (e.g. a residual ADD several
        # layers downstream, referencing it by its original un-suffixed
        # graph_idx) -- their tensor objects need the exact same start-at-0
        # treatment, or the allocator thinks they're only live from their own
        # (much later) position and overlaps their address with something
        # still genuinely in use during patch processing. Collect every
        # boundary tensor's base graph_idx once so the per-tensor loop below
        # can apply the override regardless of which op owns the tensor.
        patch_boundary_base_ids = {
            _base_graph_idx(boundary_op.input_tensors[0].graph_idx)
            for boundary_op in self.layer
            if boundary_op.params.get("is_start_of_normal_inference_block")
        }

        # Every EXISTING aliasing case above (DEPTHWISE_CONV_2D, the
        # TRANSPOSE_CONV_2D special case, MUL's self-gate, ADD's residual
        # reuse) only ever merges same-size tensors, so addTensor() below
        # using whichever tensor happens to be processed first's own len()
        # has always been correct by construction. The new CONV_2D rule
        # above merges a SMALLER output onto a (possibly already-oversized,
        # chained) input -- if that slot's size were taken from whichever
        # tensor claims it first, a later, genuinely-bigger member of the
        # same chain could silently get an undersized allocation. Computed
        # and used only under the same flag as that new rule, so every
        # existing (same-size-only) case gets byte-identical sizes either way.
        max_len_by_idx = {}
        if self.ENABLE_POINTWISE_OVERWRITE:
            for o in self.layer:
                for t in list(o.input_tensors) + list(o.output_tensors):
                    k = _base_graph_idx(t.graph_idx)
                    tl = t.len()
                    if k not in max_len_by_idx or tl > max_len_by_idx[k]:
                        max_len_by_idx[k] = tl

        # go through all tensors in the model
        for i, op in enumerate(self.layer):
            # get all unallocated tensors for this layer
            unallocated_tensors = []
            for t in op.input_tensors:
                if t.allocator_idx is None:
                    unallocated_tensors.append(t)
            for cnt, t in enumerate(op.output_tensors):
                if (
                    cnt == 0
                    and not (
                        self.USE_INPLACE
                        and op.params["op"] == "DEPTHWISE_CONV_2D"
                        and op.params["input_dtype"] == "int8"
                        and not self.tflite_op
                    )
                    and not (
                        self.USE_INPLACE
                        and op.params["op"] == "TRANSPOSE_CONV_2D"
                        and op.params["group"] == op.params["input_c"]
                        and op.params["group"] == op.params["output_c"]
                        and not self.tflite_op
                        and op.params["stride_h"] == 1
                        and op.params["stride_w"] == 1
                    )
                    and not (
                        self.ENABLE_POINTWISE_OVERWRITE
                        and op.params["op"] == "CONV_2D"
                        and op.params["input_dtype"] == "int8"
                        and not self.tflite_op
                        and op.params.get("kernel_h") == 1
                        and op.params.get("kernel_w") == 1
                        and op.params.get("stride_h") == 1
                        and op.params.get("stride_w") == 1
                        and op.params["output_c"] < op.params["input_c"]
                    )
                ):
                    if t.allocator_idx is None:
                        unallocated_tensors.append(t)
                # assume seocnd outputs will not be inplace updated
                else:
                    if t.allocator_idx is None:
                        unallocated_tensors.append(t)

            # add each tensor
            training_start_idx = _find_training_idx(layers=self.layer)
            for cnt, t in enumerate(unallocated_tensors):
                start_idx = i
                # TODO: this is temp solution
                if training_start_idx > i and "out_multiply" not in t.graph_idx:
                    end_idx = i + 1 if i == 0 else num_layers
                else:
                    end_idx = i + 1
                for idx in range(start_idx + 1, num_layers):
                    for input_t in self.layer[idx].input_tensors:
                        if _base_graph_idx(t.graph_idx) == _base_graph_idx(input_t.graph_idx):
                            end_idx = idx + 1
                # check if this is output
                ttype = TTYPE_INFERNECE
                if self.outputTables is not None and not FUSHION_CONFIG[FUSE_SGD_UPDATE_STR]:
                    for o in self.outputTables:
                        if o.idx in t.graph_idx:
                            end_idx = len(self.layer)
                            all_t_size += o.len
                            ttype = TTYPE_TRAINING_GRADIENT

                # for patchbased inference, we need the input tensor to be allocated in the patch inference stage --
                # this applies to EVERY NORMAL-SIDE consumer of a patch-boundary tensor, not just the one op
                # InputResizer.py happened to tag (see patch_boundary_base_ids comment above). Must NOT apply to
                # the patch side's own ops (is_patch=True): last_patch_op's own output tensor naturally shares this
                # same base graph_idx (it's literally what gets scattered into the reassembled tensor), but it's a
                # small, per-patch-local scratch buffer reused across all patches, not the big reassembled one --
                # forcing ITS lifetime to start at 0 too made the allocator collide the two at the same address
                # (confirmed via a QEMU run where output_ptr and patch_output ended up pointing at the same byte).
                if _base_graph_idx(t.graph_idx) in patch_boundary_base_ids and not op.get_layer_info().get("is_patch"):
                    start_idx = 0
                # add the tensor
                tensor_size = max_len_by_idx.get(_base_graph_idx(t.graph_idx), t.len()) if self.ENABLE_POINTWISE_OVERWRITE else t.len()
                t.allocator_idx = self.allocator.addTensor(start_idx, end_idx, tensor_size, name=t.graph_idx, type=ttype)
                # propagate the allocation to tensors with the same idx
                for j in range(i + 1, num_layers):
                    opp = self.layer[j]
                    for tt in opp.input_tensors:
                        if str(t.graph_idx) == str(tt.graph_idx):
                            tt.allocator_idx = t.allocator_idx
                    # not inplace update
                    for tt in opp.output_tensors:
                        if str(t.graph_idx) == str(tt.graph_idx):
                            tt.allocator_idx = t.allocator_idx

            # for detailed memory
            layermem = {}

            layermem["MAC"] = op.get_macs()
            layermem["activation"] = op.get_activation_size()
            # get_scale_size() reports what a layer *could* need for the
            # float-scales "_fpreq" kernel family -- CONV_2D/DEPTHWISE_CONV_2D
            # only actually emit and reference that array when fp_requantize
            # is on; otherwise CodeGenerator never writes it (see its
            # _parseTrainable(), which mirrors this exact condition), so
            # counting it here would overstate real flash. STAR_FORWARD
            # always needs its scales regardless of this flag (its kernel has
            # no int-multiplier/shift path at all).
            if self.fp_requantize or op.params.get("op") == "STAR_FORWARD":
                layermem["scale"] = op.get_scale_size()
            else:
                layermem["scale"] = 0
            layermem["runtime"] = op.get_sbuf_size()
            layermem["kernel"] = op.get_kbuf_size()
            self._enlargeBuffer("im2col", layermem["runtime"])
            self._enlargeBuffer("kernel", layermem["kernel"])

            if (
                "weight_name" in op.params
                and self._isTranable(op.params["weight_name"])
                and op.params["op"] != "TRANSPOSE_CONV_2D"
            ):
                size = int(op.get_weights_size())
                self.buffers["trainable"] += size
                layermem["trainable"] = size
                layermem["weight"] = 0
            else:
                layermem["weight"] = int(op.get_weights_size())
            if "bias_name" in op.params and self._isTranable(op.params["bias_name"]):
                size = int(op.get_bias_size())
                self.buffers["trainable"] += size
                if "trainable" in layermem:
                    layermem["trainable"] += size
                else:
                    layermem["trainable"] = size
                layermem["bias"] = 0
            else:
                layermem["bias"] = int(op.get_bias_size())
            # if it is float32 op, then their wegiths/bias should from SRAM buffers
            if op.params["input_dtype"] != "int8":
                layermem["scale"] = 0
                layermem["bias"] = 0
                layermem["weight"] = 0
            self.__increaseFlash(layermem["weight"])
            self.__increaseFlash(layermem["bias"])
            self.__increaseFlash(layermem["scale"])

            self.layermem.append(layermem)

        # assign data dtype for each tensor for visualization
        # we need to figure out training_weight and training_activation here
        # for training_weight, it should contain weights of "transpose conv"
        # then, other tensors in training can be categorized as training activation
        training_start_idx = _find_training_idx(self.layer)
        # assign every tenosrs labeled as TTYPE_INFERNECE after the index as TTYPE_TRAINING_ACTIVATION
        for r in self.allocator.rectangles:
            if r["type"] == TTYPE_INFERNECE and r["end"] > training_start_idx:
                r["type"] = TTYPE_TRAINING_ACTIVATION
        # for each tranpose conv, find it
        for i, op in enumerate(self.layer):
            if op.params["op"] == "TRANSPOSE_CONV_2D":
                # if any tenosr used by this layer
                for r in self.allocator.rectangles:
                    if r["end"] <= training_start_idx:
                        continue
                    if r["name"] == op.params["weight_name"]:
                        r["type"] = TTYPE_TRAINING_WEIGHTS

        # find out int8 inplace depthwise conv and stride == 2
        for i, op in enumerate(self.layer):
            if (
                op.params["op"] == "DEPTHWISE_CONV_2D"
                and op.params["input_dtype"] == "int8"
                and op.params["stride_h"] == op.params["stride_w"] == 2
            ):
                if op.input_tensors[0].allocator_idx == op.output_tensors[0].allocator_idx:
                    self.allocator.rectangles[op.input_tensors[0].allocator_idx]["stride2_inplace_idx"] = i

        # Reorder the rectangles to decide which tensor needs to be scheduled first
        self.allocator.sortSize()
        self.allocator.allocate()
        if self.ENABLE_POINTWISE_OVERWRITE:
            # Tighten first, on the untouched base schedule, before
            # _reclaimPointwiseGrowthSpace introduces pairs whose relative
            # offset (output_base vs input_base) must stay exact -- a later
            # independent, per-rectangle compaction pass could slide one
            # member of such a pair without its partner and reintroduce an
            # unsafe overlap. Running compaction only here, before any pair
            # exists, sidesteps that risk entirely.
            self._compactSchedule()
            self._reclaimPointwiseGrowthSpace()
            self._reclaimStemConvOverlap()
        self.allocator.visualize(self.mem_visual_path)
        self._enlargeBuffer("input_output", self.allocator.get_peak())

        # sanity check, see if all tensors have been allocated
        for i, op in enumerate(self.layer):
            # get all unallocated tensors for this layer
            for cnt, t in enumerate(op.input_tensors):
                assert t.allocator_idx is not None
            for cnt, t in enumerate(op.output_tensors):
                assert t.allocator_idx is not None

        # assign the address according to placement
        for i, op in enumerate(self.layer):
            # get all unallocated tensors for this layer
            for cnt, t in enumerate(op.input_tensors):
                if cnt == 0:
                    op.params["input_buf_add_offset"] = self.allocator.getIdxAddress(t.allocator_idx)
                    op.params["input_buf_add"] = "front"
                elif cnt == 1:
                    op.params["input2_buf_add_offset"] = self.allocator.getIdxAddress(t.allocator_idx)
                    op.params["input2_buf_add"] = "front"
                elif cnt == 2:
                    op.params["input3_buf_add_offset"] = self.allocator.getIdxAddress(t.allocator_idx)
                    op.params["input3_buf_add"] = "front"
                op.input_tensors[cnt].buffer_name = "buffer0"
                op.input_tensors[cnt].buffer_address = self.allocator.getIdxAddress(t.allocator_idx)
            for cnt, t in enumerate(op.output_tensors):
                if cnt == 0:
                    op.params["output_buf_add_offset"] = self.allocator.getIdxAddress(t.allocator_idx)
                    op.params["output_buf_add"] = "front"
                    op.output_tensors[cnt].buffer_name = "buffer0"
                    op.output_tensors[cnt].buffer_address = self.allocator.getIdxAddress(t.allocator_idx)
                if cnt == 1:
                    op.params["output2_buf_add_offset"] = self.allocator.getIdxAddress(t.allocator_idx)
                    op.params["output2_buf_add"] = "front"
                    op.output_tensors[cnt].buffer_name = "buffer0"
                    op.output_tensors[cnt].buffer_address = self.allocator.getIdxAddress(t.allocator_idx)

        # calculate peak mem
        self.peakmem = (
            self.allocator.get_peak() + self.buffers["im2col"] + self.buffers["kernel"]  # + self.buffers["trainable"]
        )

    def dumpLayerIndex(self):
        # header
        print("-" * 14 + " Tensor Allocation Details " + "-" * 14)
        print(" #op |   operator type   | input index | output index |")
        for cnt, l in enumerate(self.layer):
            operator_num = "#" + str(cnt)
            type = str(l.params["op"])
            input_tensor = ""
            for cnt_inp, inp in enumerate(l.input_tensors):
                input_tensor += str(inp.allocator_idx)
                if cnt_inp < len(l.input_tensors) - 1:
                    input_tensor += ","
            output_tensor = str(l.output_tensors[0].allocator_idx)
            string = (
                operator_num.ljust(5)
                + "|"
                + type.ljust(19)
                + "|"
                + input_tensor.ljust(13)
                + "|"
                + output_tensor.ljust(14)
                + "|"
            )
            print(string)

    def dumpLayerMem(self):
        # header
        print(
            "----------------------------------------------------  Schedule Details ----------------------------------------------------------------"  # noqa: E501
        )
        print(
            "----------------------|                      SRAM                      ||                     Flash                      |             |"  # noqa: E501
        )
        print(
            "----------------------|  activation  |  runtime  | trainable  |  sum   ||   weight   |   bias   |  scale   |     sum     |     MAC     |"  # noqa: E501
        )

        layermem = self.layermem
        self.__dumpMemInfo(layermem)

    def __dumpMemInfo(self, layermem):
        string = "-------Schedule-------|"
        maxActive = self.buffers["input_output"]
        maxRuntime = self.buffers["im2col"] + self.buffers["kernel"]
        maxTrainable = self.buffers["trainable"]
        totalWeight = self.__sumKey(layermem, "weight")
        totalBias = self.__sumKey(layermem, "bias")
        totalScale = self.__sumKey(layermem, "scale")
        totalMAC = self.__sumKey(layermem, "MAC")
        string += str(maxActive).ljust(14) + "|"
        string += str(maxRuntime).ljust(11) + "|"
        string += str(maxTrainable).ljust(12) + "|"
        string += str(maxActive + maxRuntime + maxTrainable).ljust(8) + "||"
        string += str(totalWeight).ljust(12) + "|"
        string += str(totalBias).ljust(10) + "|"
        string += str(totalScale).ljust(10) + "|"
        string += str(totalWeight + totalBias + totalScale).ljust(13) + "|"
        string += str(totalMAC).ljust(13) + "|"
        print(string)
        for i, _ in enumerate(layermem):
            layer_info = self.layer[i].get_layer_info()
            string = ""
            string += str(i) + ":" + layer_info["op"]
            string = string.ljust(22) + "|"
            SRAM = 0
            if "activation" in layermem[i]:
                substr = (
                    str(layermem[i]["activation"]) + " (" + "{:.0%}".format(layermem[i]["activation"] / maxActive) + ")"
                )
                string += substr.ljust(14) + "|"
                SRAM += layermem[i]["activation"]
            if "runtime" in layermem[i]:
                sbuf = layermem[i]["runtime"] + layermem[i]["kernel"]
                substr = str(sbuf) + " (" + "{:.0%}".format(sbuf / maxRuntime) + ")"
                string += substr.ljust(11) + "|"
                SRAM += sbuf
            else:
                string = string.ljust(49) + "|"
            if "trainable" in layermem[i]:
                substr = (
                    str(layermem[i]["trainable"])
                    + " ("
                    + "{:.0%}".format(layermem[i]["trainable"] / maxTrainable)
                    + ")"
                )
                string += substr.ljust(12) + "|"
                SRAM += layermem[i]["trainable"]
            else:
                string = string.ljust(62) + "|"

            # SRAM end
            string += str(SRAM)
            string = string.ljust(71) + "||"
            flash = 0
            if "weight" in layermem[i]:
                substr = (
                    str(layermem[i]["weight"])
                    + " ("
                    + "{:.0%}".format(layermem[i]["weight"] / (totalWeight + 0.0001))
                    + ")"
                )
                string += str(substr).ljust(12) + "|"
                flash += layermem[i]["weight"]
            if "bias" in layermem[i]:
                substr = (
                    str(layermem[i]["bias"]) + " (" + "{:.0%}".format(layermem[i]["bias"] / (totalBias + 0.0001)) + ")"
                )
                string += str(substr).ljust(10) + "|"
                flash += layermem[i]["bias"]
            if "scale" in layermem[i]:
                substr = (
                    str(layermem[i]["scale"]) + " (" + "{:.0%}".format(layermem[i]["scale"] / totalScale + 0.0001) + ")"
                )
                string += str(substr).ljust(10) + "|"
                flash += layermem[i]["scale"]

                if flash > 0:
                    string += (
                        str(flash)
                        + " ("
                        + "{:.0%}".format(flash / (totalWeight + totalBias + totalScale + 0.0001))
                        + ")"
                    )
                    string = string.ljust(121) + "|"
            # flash end
            if "MAC" in layermem[i]:
                substr = str(layermem[i]["MAC"]) + " (" + "{:.0%}".format(layermem[i]["MAC"] / totalMAC) + ")"
                string += str(substr).ljust(13) + "|"
            print(string)

    def __sumKey(self, layers, key):
        result = 0
        for _, layer in enumerate(layers):
            if key in layer:
                result += layer[key]

        return result

    def getBuffers(self):
        return self.buffers

    # Maximum binary size: This should be updated if any change in the inference side
    # TODO: Combine with code generation to get more accurate result
    def profileResult(self):
        return self.peakmem, self.flash + self.bias + self.scale + int(self.code * 1024)

    def __increaseFlash(self, size):
        self.flash += int(size)

    def _enlargeBuffer(self, buf_str, size):
        if buf_str == "input_output" or buf_str == "residual":
            self.buffers[buf_str] = max(self.buffers[buf_str], int(size))
        else:
            if buf_str not in self.buffers:
                self.buffers[buf_str] = size
            else:
                self.buffers[buf_str] = max(self.buffers[buf_str], size)


def _find_training_idx(layers):
    idx = len(layers)
    for cnt, l in enumerate(layers):
        if l.params["op"] in ["CAST"]:
            return cnt
    return idx
