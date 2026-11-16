"""Convert an Imspector OBF/MSR file into one ``ArrayDataset`` per acquisition.

Each OBF *acquisition* becomes a pyramided ``ArrayDataset``. Facts true of the whole
acquisition (the OME record, the microscope body, the stage, the OBF stack headers) hang
off the origin anchor; facts true of one channel (histogram, name, detector, light path)
hang off that channel's anchor. Physical scale is an edge, so every calibrated
acquisition registers into one shared micrometer world with a single scene over it,
**one layer per channel** — the only exception being a packed RGB(A) photograph, decided
from the pixel type in `load_stack` and from nothing else.

Four things are specific to OBF, and are where this differs from its OME sibling.

**A stack is not an image.** Imspector writes one single-channel stack per dye —
``Overview 8/Image 3/17:28:01/STAR RED`` — and writes its own maximum projections back
into the file beside them, so a four-channel z-stack arrives as eight stacks. Stacks are
grouped by the path above the dye, derived stacks dropped, and the rest concatenated
along the channel axis. Each merged channel carries the metadata of *its own* source
stack's OME image rather than reading them all off one.

**The OME record is optional and need not correspond.** It is matched per stack by name
then by position, and every consumer accepts ``None``: a file with no OME block still
converts, with a fallback palette and no light path.

**A length in the stack header is only a length where the file says so.** The header
carries ``lengths``/``offsets`` for every dimension in metres, in OBF's reversed axis
order. Dividing blindly gives the channel axis a pixel size of one metre; the footer's
per-dimension SI units are the guard — see `si_length_factor`.

**The header's offsets are not stage positions.** Every axis carries the same ``-0.5``, a
normalised default. Stage position comes from the OME planes or from nowhere; the raw
offsets are recorded on the ``obf-stack`` device state instead.

One reader note: `obffile.ObfStack.asarray` decompresses a chunked stack into a temporary
on-disk Zarr living as long as the array does. This converter holds one stack at a time,
so eight multi-gigabyte stacks do not accumulate eight scratch directories.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any

import xarray as xr
from arkitekt import App, DescriptorOperator, Requires, Task, run
from kanne.scalars import Unit
from mikro import Mikro, build_pyramid, canonical, mikro_service, scales_from, space_3d
from mikro.api.schema import (
    PlacementValidity,
    ArrayDataset,
    AxisAnchorInput,
    Blending,
    ColorMap,
    CoordinateAnchorInput,
    DeviceStateInput,
    File,
    LabelInput,
    OmeMetadataInput,
    OptikitStateInput,
    PhasorColorMode,
    PhasorTransferInput,
    PreferredView,
    ProjectionMode,
    Scene,
    SourceFileInput,
    StageStateInput,
    ValueHistogramInput,
    ValueRelation,
)
from mikro.arkitekt.specs import EXTENSION
from mikro.inputs.render import (
    RGB_PRIMARIES,
    color_for_wavelength,
    composite_graph,
    fallback_color,
    rgba,
)
from mikro.pyramid import axes_for

from objreader.converter_metadata import device, devices, setting, text

# The package's forked `obffile.py`, not the PyPI package: upstream returns a plain array,
# this one streams a decompressed stack into a scratch Zarr and returns a dask array.
from objreader.obffile import ObfFile
from objreader.ome_metadata import (
    COLOR_BY_EMISSION,
    FAR_RED,
    channel_lightpath,
    channel_optikit,
    find_instrument,
    find_objective,
    ome_rgba,
    qty,
    to_micrometer,
    um,
    unit_symbol,
)

__version__ = "0.1.3"

logger = logging.getLogger(__name__)

app = App(
    "objreader",
    __version__,
    description=(
        "Converts Imspector OBF/MSR files into mikro datasets: one pyramided, "
        "multi-channel dataset per acquisition, with its OME metadata, light path and "
        "device state, placed in a shared micrometer space."
    ),
    services=[mikro_service],
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The axes the world is built over. Scales and offsets come from intersecting a stack's
# dims with this set, never from the OME fields: `register` raises on an axis the two
# spaces do not share, and a projection stack routinely carries a populated `position_z`.
WORLD_AXES = frozenset({"z", "y", "x"})

# The axis names the library's bare-name convention knows, and so the only ones this
# converter may leave undeclared. `load_stack` types every dim outside this set, and
# `default_axis_type` raises `UnknownAxisName` on anything else — it used to call an
# unrecognised name SPACE silently, and the pyramid coarsens exactly the SPACE axes, so a
# footer label like `rep` was halved away per level with nothing said.
KNOWN_AXIS_NAMES = frozenset({"t", "c", "z", "y", "x"})

AXIS_LONG_NAMES = {
    "t": "time",
    "c": "channel",
    "z": "depth",
    "y": "height",
    "x": "width",
}

# The dimension names OBF spells in upper case and this converter renames. Held as a set
# rather than tested with `in "TCZYXS"`, which is a substring match and would also
# accept a footer label of "TC" or "ZY".
OBF_AXIS_NAMES = frozenset({"T", "C", "Z", "Y", "X"})

# Above this total extent a seconds-measured axis is frame time, below it photon arrival
# time. A microsecond is not a boundary anything sits near: one laser period is ~12.5 ns
# at 80 MHz, and the fastest timelapse worth calling one is milliseconds a frame. See
# `microtime_axis_of`.
MICROTIME_MAX_EXTENT_S = 1e-6

# `ValueHistogramInput.from_array` calls `np.asarray`, the one place a lazy stack is
# forced into memory. Above this many bytes per channel slice the histogram comes off the
# coarsest pyramid level instead. Reached routinely: one channel of a 189-timepoint
# 10-plane 912x857 int16 stack is 2.9 GB.
HISTOGRAM_BUDGET_BYTES = 256 * 1024 * 1024

# Past this many layers a scene stops being useful, and one layer per channel reaches it
# easily. The world holds every registration and outlives the run, so a subset can be
# staged later.
MAX_SCENE_LAYERS = 48

# ---------------------------------------------------------------------------
# Units and value coercion
#
# Quantity fields are pint-backed and reject a bare float; a None slipping through
# string formatting becomes the literal "None nm". Both are closed here rather than at
# ~60 call sites: these helpers return None when they have nothing to say.
# ---------------------------------------------------------------------------


def micrometers(value: float | None) -> str | None:
    """A number already in micrometers, as a Length scalar string.

    The counterpart to :func:`um` for the OBF header, whose lengths have been converted
    from metres by hand and so arrive without an OME unit to interpret.
    """
    if value is None or not math.isfinite(value):
        return None
    return f"{value:g} um"


# ---------------------------------------------------------------------------
# The OBF stack header
#
# OBF stores per-dimension geometry in the file's own axis order (dimension 0 is X) while
# `ObfStack.dims` is reversed into C order. Everything goes through `stack_axis_indices`
# rather than zipping positionally: a reversal error does not fail, it silently swaps the
# x and z pixel sizes.
# ---------------------------------------------------------------------------


def stack_axis_indices(stack: Any) -> dict[str, int]:
    """Axis name (lower case, as this converter names it) -> OBF dimension index.

    Empty when the two cannot be lined up with certainty: ``squeeze=True`` drops length-1
    dimensions from ``sizes`` while leaving ``header.res`` at full rank, and guessing
    which header entry survives would produce a plausible, wrong calibration. This
    converter always opens with ``squeeze=False``; the check keeps that guarantee local.
    """
    names = [str(d) for d in stack.dims]
    # A trailing "S" is the RGB/RGBA sample axis. It is synthesised by the reader from
    # the pixel type, not stored as an OBF dimension, so it has no header entry.
    if names and names[-1] == "S":
        names = names[:-1]

    rank = int(stack.header.rank)
    if len(names) != rank:
        logger.warning(
            "Stack %r exposes %s dims for a rank-%s header; not reading any header "
            "geometry from it.",
            stack.name,
            len(names),
            rank,
        )
        return {}

    # `dims` is `header.res` reversed, so index k from the front is index rank-1-k
    # from the back.
    return {name.lower(): rank - 1 - k for k, name in enumerate(names)}


def si_length_factor(unit: Any) -> float | None:
    """The factor turning a header value under `unit` into metres, if it is a length.

    ``ObfSiUnit`` is a full rational-exponent SI record, so "is this a length" has an
    exact answer: metres to the first power, every other base unit to the zeroth.
    Anything else — the dimensionless unit OBF gives a channel axis, the seconds it gives
    a time axis — returns None and is dropped. Without this guard ``lengths[i] / res[i]``
    makes a channel axis one metre per channel, and a projection's collapsed z axis 1e6 µm.
    """
    if unit is None:
        return None
    exponents = (
        unit.meters,
        unit.kilograms,
        unit.seconds,
        unit.amperes,
        unit.kelvin,
        unit.moles,
        unit.candela,
        unit.radian,
        unit.steradian,
    )
    if tuple(exponents[0]) != (1, 1):
        return None
    if any(numerator != 0 for numerator, _ in exponents[1:]):
        return None
    factor = float(unit.scale_factor)
    if not math.isfinite(factor) or factor <= 0:
        return None
    return factor


def si_time_factor(unit: Any) -> float | None:
    """Seconds per unit for an axis the footer measures in time, else None.

    The sibling of :func:`si_length_factor` — seconds to the first power, every other
    base unit to the zeroth — picking up the axes that function deliberately drops.
    """
    if unit is None:
        return None
    exponents = (
        unit.meters,
        unit.kilograms,
        unit.seconds,
        unit.amperes,
        unit.kelvin,
        unit.moles,
        unit.candela,
        unit.radian,
        unit.steradian,
    )
    if tuple(exponents[2]) != (1, 1):
        return None
    if any(numerator != 0 for index, (numerator, _) in enumerate(exponents) if index != 2):
        return None
    factor = float(unit.scale_factor)
    if not math.isfinite(factor) or factor <= 0:
        return None
    return factor


def microtime_axis_of(stack: Any) -> tuple[str, float] | None:
    """The stack's photon-arrival axis and its total extent in seconds, or None.

    A phasor is a transform over a *continuous* axis, and the one OBF carries is FLIM
    arrival time. Nothing labels it, so the signal is the file's own declared SI unit: an
    arrival-time axis is measured in seconds.

    Seconds alone is not enough — a timelapse's frame axis is seconds too. What separates
    them is magnitude, by orders of magnitude: a microtime axis spans one laser period
    (12.5 ns at 80 MHz) where frames are seconds apart. The cut is at a microsecond, a
    thousand times either side of both. A singleton is refused: one arrival-time bin is
    not a decay.

    Returns ``(axis name, extent in seconds)`` — the extent so a misfire on some future
    file is visible as a number rather than inferred from a layer that should not exist.
    """
    # Tolerant of a stack that exposes neither, the way every other header read here is:
    # an OBF file written by Imspector carries both, an OBF stack in general need not, and
    # "this file declares no units" is an answer rather than a crash.
    header = getattr(stack, "header", None)
    footer = getattr(stack, "footer", None)
    if header is None or footer is None:
        return None
    si_units = list(getattr(footer, "si_dimensions", []) or [])

    for name, index in stack_axis_indices(stack).items():
        if index >= len(header.res) or index >= len(header.lengths):
            continue
        factor = si_time_factor(si_units[index] if index < len(si_units) else None)
        if factor is None:
            continue

        resolution = int(header.res[index])
        length = float(header.lengths[index])
        if resolution <= 1 or not math.isfinite(length) or length <= 0:
            continue

        extent = length * factor
        if extent < MICROTIME_MAX_EXTENT_S:
            return name, extent

    return None


def header_voxel_sizes(stack: Any) -> dict[str, float]:
    """Pixel size in micrometers per axis, from the stack header.

    Only for the axes the footer calls a length; see :func:`si_length_factor`. A zero or
    non-finite extent is dropped too: a length of zero is an unset field, not a small pixel.
    """
    header = stack.header
    si_units = list(getattr(stack.footer, "si_dimensions", []) or [])

    sizes: dict[str, float] = {}
    for name, index in stack_axis_indices(stack).items():
        if index >= len(header.res) or index >= len(header.lengths):
            continue
        factor = si_length_factor(si_units[index] if index < len(si_units) else None)
        if factor is None:
            continue

        resolution = int(header.res[index])
        length = float(header.lengths[index])
        if resolution <= 0 or not math.isfinite(length) or length <= 0:
            continue

        sizes[name] = (length * factor / resolution) * 1e6
    return sizes


def obf_stack_device(stack: Any, index: int) -> DeviceStateInput | None:
    """The stack header, verbatim, as a device state.

    Everything OBF records about the stack's geometry, kept whether or not this converter
    acted on it. The offsets are here *because* they are refused as stage positions —
    recording them as text keeps the file's own words available without turning a
    normalised -0.5 into a placement.
    """
    header = stack.header
    footer = stack.footer
    rank = int(header.rank)
    si_units = list(getattr(footer, "si_dimensions", []) or [])
    tags = dict(getattr(footer, "tag_dictionary", {}) or {})

    return device(
        f"obf-stack-{index}",
        "acquisition",
        setting("stack-name", string=text(stack.name)),
        setting("stack-index", number=float(index)),
        setting("stack-description", string=text(header.description)),
        setting("format-version", number=float(header.format_version)),
        setting("rank", number=float(rank)),
        setting("resolution", string=str(list(header.res[:rank]))),
        setting("lengths-m", string=str([float(v) for v in header.lengths[:rank]])),
        setting("offsets-m", string=str([float(v) for v in header.offsets[:rank]])),
        setting("si-dimensions", string=str([str(u) for u in si_units[:rank]])),
        setting("si-value", string=text(getattr(footer, "si_value", None))),
        setting("dimension-labels", string=str(list(getattr(footer, "labels", []) or []))),
        setting("array-dims", string=str([str(d) for d in stack.dims])),
        setting("array-shape", string=str(list(stack.shape))),
        setting("dtype", string=str(stack.dtype)),
        setting("compression-type", number=float(header.compression_type)),
        setting("compression-level", number=float(header.compression_level)),
        setting("chunk-count", number=float(len(getattr(footer, "chunk_positions", []) or []))),
        setting("data-length", number=float(header.data_length)),
        # Imspector's own tag blobs. Names only: the values are unparsed XML/JSON
        # documents, and the whole of the file's structured metadata is already carried
        # as the OME record on this same anchor.
        setting("tag-dictionary-keys", string=str(sorted(tags)) if tags else None),
        setting("legacy-metadata", string=text(getattr(footer, "metadata", None))),
    )


# ---------------------------------------------------------------------------
# The array
# ---------------------------------------------------------------------------


def load_stack(
    stack: Any, tile: int | None = None
) -> tuple[xr.DataArray, dict[str, str], bool]:
    """One OBF stack as a named array, the axis types to state, and is-it-RGB.

    Returns the array with its dims reduced to the ones this converter will claim, a
    ``{name: TYPE}`` mapping for any axis the naming convention would get wrong, and
    whether the ``c`` axis is a *sample* axis — 3 or 4 samples packed into one pixel.

    That last flag is the file's own answer to "is this RGB" and the only trustworthy
    one. It is decided here, where the reader's synthesised "S" axis is still visible;
    three lines down it has been renamed to ``c`` and is indistinguishable from three
    dyes. Every alternative test is a guess — see `merge_members`.

    With ``tile``, only that index of the outermost axis is read and that axis is gone
    from the result. That is the mosaic path: one contiguous run of the decompressed
    stream instead of all of it, and `ObfStack.read_outer` returns an eager array, which
    keeps `build_pyramid` from writing a scratch store of its own.

    The policy on dims is the OME converter's: keep what has extent, declare what is
    unrecognised. Expanding every stack to a full 5D would make a singleton ``t`` assert
    a time series and a singleton ``z`` a slab thickness the file never measured.
    """
    if tile is None:
        array = xr.DataArray(stack.asarray(), dims=[str(d) for d in stack.dims])
    else:
        # One index of the outermost axis, read on its own. `read_outer` returns an
        # eager ndarray of the remaining dims, so the outer name drops out of `dims`
        # with it — see `ObfStack._outer_layout`.
        dims = [str(d) for d in stack.dims[1:]]
        array = xr.DataArray(stack.read_outer(tile), dims=dims)

    # RGB samples. The reader appends an "S" axis for a pixel type that packs three or
    # four samples per pixel; selecting sample 0 (the obvious reading) throws away two
    # thirds of the picture.
    #
    # `packed_samples` is recorded here and nowhere else. Only 3 samples (RGB) or 4
    # (RGBA) are a picture: any other extent behind an "S" axis is some other packing,
    # and the honest reading of it is ordinary channels — one layer each.
    packed_samples = False
    if "S" in array.dims:
        if array.sizes["S"] > 1 and array.sizes.get("C", 1) == 1:
            packed_samples = array.sizes["S"] in (3, 4)
            if "C" in array.dims:
                array = array.squeeze("C", drop=True)
            array = array.rename({"S": "c"})
        else:
            array = array.isel(S=0, drop=True)

    # Only the five names OBF spells in upper case, and only where the lower-case name
    # is still free: a footer label of "x" alongside a positional "X" would otherwise
    # collide into one dimension and take the array's shape with it.
    present = {str(d) for d in array.dims}
    renames = {
        str(d): str(d).lower()
        for d in array.dims
        if str(d) in OBF_AXIS_NAMES and str(d).lower() not in present
    }
    if renames:
        array = array.rename(renames)

    if "c" not in array.dims:
        array = array.expand_dims("c")

    # Keep c, y and x unconditionally; keep t and z only where they have extent.
    #
    # A kept axis is a permanent structural claim — the client validator requires the
    # dim set and the `axes=` list to match exactly, so there is no way to carry an axis
    # quietly. A singleton `t` asserts a time series and hands the UI a slider with one
    # stop; worse, a singleton `z` makes the world willing to accept a z scale, and a
    # maximum projection has no slab thickness to state. This matters more for OBF than
    # for its sibling, because opening with `squeeze=False` means every stack arrives at
    # full rank with its singletons intact.
    #
    # `c` is kept even at extent 1 for the opposite reason: every per-channel anchor
    # keys off `AxisAnchorInput(axis="c", ...)` and every channel render graph
    # composites along a CHANNEL axis, so a single-channel stack without one has nowhere
    # to hang its histogram, its label, its detector or its light path. OBF files are
    # routinely one channel per stack, so this is the common case here, not the corner.
    dropped = [
        d for d in array.dims if str(d) not in ("c", "y", "x") and array.sizes[d] == 1
    ]
    if dropped:
        array = array.squeeze(dropped, drop=True)

    # Anything left that is not t/c/z/y/x came from a footer label or from a rank above
    # five. It has to be declared: the type default calls it SPACE, which would both
    # sort it in among z/y/x and hand it to the pyramid to be halved.
    extra_types = {str(d): "INDEX" for d in array.dims if str(d) not in KNOWN_AXIS_NAMES}

    # A FLIM arrival-time axis, if the footer's own SI units say there is one. Declared
    # over the top of whatever the line above or the bare-name convention decided, and
    # that override is the point: Imspector writes this axis under an ordinary name, so
    # left alone it becomes SPACE (if it landed on `z`), TIME (on `t`) or INDEX, and all
    # three are wrong in the same way -- a continuous physical axis called something a
    # phasor cannot be taken over.
    #
    # Matched on the lower-cased name because `stack_axis_indices` lower-cases and a
    # footer label need not: the array may still be carrying the label's own spelling.
    microtime = microtime_axis_of(stack)
    if microtime is not None:
        name, extent = microtime
        by_lower = {str(d).lower(): str(d) for d in array.dims}
        axis = by_lower.get(name)
        if axis is not None:
            extra_types[axis] = "MICROTIME"
            logger.info(
                "Stack %r: axis %r is measured in seconds and spans %.3g s, which is "
                "photon arrival time rather than frame time; typing it MICROTIME.",
                stack.name,
                axis,
                extent,
            )

    return (
        xr.DataArray(array.data, dims=[str(d) for d in array.dims]),
        extra_types,
        packed_samples,
    )


# ---------------------------------------------------------------------------
# OME lookup
#
# Everything here tolerates a missing OME record. An OBF file written by Imspector
# carries one; an OBF file in general need not, and nothing in the format ties the
# number of stacks to the number of OME images.
# ---------------------------------------------------------------------------


def try_parse_ome(obf: Any) -> Any | None:
    """The OME document embedded in the file, or None.

    Two fallbacks: the tag key is matched case-insensitively, and failing that any value
    that looks like an OME XML document is tried. `ome_types` is imported locally so its
    absence degrades to "no OME metadata" rather than taking the app's registration down.
    """
    try:
        from ome_types import from_xml
    except ImportError:
        logger.warning("ome_types is not installed; reading the file without metadata")
        return None

    metadata = dict(getattr(obf.header, "metadata", {}) or {})
    if not metadata:
        return None

    candidates: list[str] = []
    for key, value in metadata.items():
        if not isinstance(value, str):
            continue
        if key.lower() == "ome_xml":
            candidates.insert(0, value)
        elif value.lstrip().startswith("<?xml") and "OME" in value[:2000]:
            candidates.append(value)

    for candidate in candidates:
        try:
            return from_xml(candidate)
        except Exception:
            logger.warning("An embedded XML block did not parse as OME", exc_info=True)
    return None


def find_matching_ome_image(index: int, name: str, ome: Any | None) -> Any | None:
    """The OME image describing this stack, or None.

    By name first — Imspector gives the image and the stack the same
    ``Overview/Image/timestamp/channel`` path — and by position only as a fallback.

    Returns None rather than raising, unlike its OME sibling: there a series without an
    image means the reader and the metadata disagree about one document. Here the OME
    block is an optional extra a valid OBF file can omit, so its absence is a fact about
    the file rather than a contradiction.
    """
    if ome is None:
        return None
    images = list(getattr(ome, "images", []) or [])
    if not images:
        return None

    for image in images:
        if text(image.name) == name:
            return image
    for image in images:
        if name and name in (text(image.id) or ""):
            return image
    if index < len(images):
        return images[index]
    return None


# ---------------------------------------------------------------------------
# Stack selection and grouping
#
# The unit this converter uploads is an *acquisition*, not a stack, and in an OBF file
# those are not the same thing. Imspector writes one single-channel stack per dye —
# `Overview 8/Image 3/17:28:01/STAR RED` — and writes its own maximum projections back
# into the same file as further stacks of the same shape. Read literally that is eight
# images for one four-channel acquisition, half of them derived data nobody asked to
# ingest. The embedded OME record mirrors the same split, one single-channel image per
# stack, so it confirms the reading rather than correcting it.
#
# Everything in this section puts that back together: drop what the file computed for
# itself, and merge what is left into one dataset per acquisition, one channel per
# source stack.
# ---------------------------------------------------------------------------

# What Imspector names a stack it derived from another one: `max(Z,Alexa 405)`. Anchored
# at both ends so a dye whose name merely contains a bracket is not mistaken for a
# projection. The list is the reducers seen in the wild plus the obvious siblings; a
# derived stack this misses is uploaded as an extra channel rather than lost, which is
# the failure worth having.
DERIVED_STACK = re.compile(
    r"^\s*(max|min|sum|mean|average|avg|std|var|median|proj|projection)\s*\(.*\)\s*$",
    re.IGNORECASE,
)


def channel_component(name: str) -> str:
    """The last segment of a stack path — the dye, in Imspector's naming."""
    return name.rsplit("/", 1)[-1].strip()


def measurement_key(name: str) -> str:
    """Everything before the dye: the acquisition this stack belongs to.

    A name with no separator is its own acquisition — the honest reading for a file
    written by anything but Imspector, where merging on no evidence would invent a
    multi-channel image out of two unrelated pictures.
    """
    head, sep, _ = name.rpartition("/")
    return head.strip() if sep else name.strip()


def is_derived(name: str) -> bool:
    """Whether this stack is something the acquisition software computed."""
    return bool(DERIVED_STACK.match(channel_component(name)))


def geometry_signature(stack: Any) -> tuple[tuple[str, int], ...]:
    """The stack's shape with the channel axis removed.

    Read off the header, not a loaded array: `ObfStack.asarray` decompresses a compressed
    stack to a scratch Zarr, and every stack in the sample file is compressed, so grouping
    on pixels would decompress the file twice over just to decide what to merge.

    Channel is excluded because merging along it is what the grouping is for; every other
    axis has to match or `xr.concat` broadcasts silently or fails.
    """
    return tuple(
        (str(dim), int(size))
        for dim, size in zip(stack.dims, stack.shape)
        if str(dim).upper() != "C"
    )


@dataclass(frozen=True)
class Member:
    """One source stack of an acquisition, with the OME record that describes it.

    The OME lookup is per member because that is how the file is written: four stacks,
    four OME images, one channel each. A merged dataset draws its channel metadata from
    four different images, so this pairing survives down to `channel_specs`.
    """

    index: int
    stack: Any
    image: Any | None
    instrument: Any | None

    @property
    def name(self) -> str:
        return text(self.stack.name) or ""


@dataclass(frozen=True)
class StackGroup:
    """One acquisition: the stacks that become the channels of a single dataset.

    The representative accessors below are member 0's, used for everything stated once per
    dataset — calibration, stage, microscope body — all facts about the acquisition rather
    than about a dye, and recorded identically on every member.
    """

    key: str
    members: list[Member]

    @property
    def stack(self) -> Any:
        return self.members[0].stack

    @property
    def index(self) -> int:
        return self.members[0].index

    @property
    def image(self) -> Any | None:
        return self.members[0].image

    @property
    def instrument(self) -> Any | None:
        return self.members[0].instrument


def group_stacks(stacks: Sequence[Any], ome: Any | None) -> list[StackGroup]:
    """The acquisitions in a file, in the order the file lists them.

    Derived stacks are dropped and the drop is logged: a silent skip is indistinguishable
    from a file that never had any. Members keep file order, so a channel's index is the
    order Imspector wrote its stacks in — stable across runs.
    """
    kept: list[tuple[int, Any, str]] = []
    skipped: list[str] = []

    for index, stack in enumerate(stacks):
        name = text(stack.name) or ""
        if is_derived(name):
            skipped.append(name)
        else:
            kept.append((index, stack, name))

    if skipped:
        logger.info(
            "Skipping %s stack(s) the file derived from others: %s",
            len(skipped),
            ", ".join(skipped),
        )

    grouped: dict[tuple[str, tuple[tuple[str, int], ...]], list[Member]] = {}
    for index, stack, name in kept:
        image = find_matching_ome_image(index, name, ome)
        key = (measurement_key(name) or f"({index})", geometry_signature(stack))
        grouped.setdefault(key, []).append(
            Member(index, stack, image, find_instrument(ome, image))
        )

    # A measurement whose stacks disagree about shape cannot become one array. It splits
    # rather than failing — the stacks are still real data — but it is named out loud,
    # because the result is two datasets where the file's naming promised one.
    by_measurement: dict[str, int] = {}
    for measurement, _ in grouped:
        by_measurement[measurement] = by_measurement.get(measurement, 0) + 1

    groups: list[StackGroup] = []
    for (measurement, signature), members in grouped.items():
        key = measurement
        if by_measurement[measurement] > 1:
            shape = "x".join(str(size) for _, size in signature)
            key = f"{measurement} [{shape}]"
            logger.warning(
                "Measurement %r holds stacks of more than one shape; %s uploaded "
                "separately as %r",
                measurement,
                len(members),
                key,
            )
        groups.append(StackGroup(key=key, members=members))

    logger.info(
        "%s stack(s) -> %s acquisition(s): %s",
        len(kept),
        len(groups),
        ", ".join(f"{g.key} ({len(g.members)} ch)" for g in groups),
    )
    return groups


def stack_metadata_json(file: File, group: StackGroup) -> str:
    """This acquisition's metadata, as the JSON string the anchor takes.

    Two records, both per *member*. The OME half is reduced to the images this
    acquisition merged rather than dumping the whole document — eight stacks would
    otherwise carry eight copies of a 2 MB document that nothing client-side reads. The
    OBF half is one stack header per member: the file's own account of its geometry,
    inexpressible in OME.

    Every member is written out, not just the representative. Each has its own header,
    compression and raw offsets, and those numbers are recorded verbatim precisely
    because this converter refused to act on some of them; keeping only the first would
    make the blob a claim rather than a record.

    Both the stack name and its index are written: ``ObfFile.stacks`` is keyed by name,
    so a name is what selected the stack, but two stacks can share one and only the index
    separates them.
    """

    def obf_record(stack: Any) -> dict[str, Any]:
        header = stack.header
        rank = int(header.rank)
        footer = stack.footer
        return {
            "format_version": header.format_version,
            "rank": rank,
            "resolution": [int(v) for v in header.res[:rank]],
            "lengths_m": [float(v) for v in header.lengths[:rank]],
            "offsets_m": [float(v) for v in header.offsets[:rank]],
            "si_dimensions": [
                str(u) for u in list(getattr(footer, "si_dimensions", []) or [])[:rank]
            ],
            "labels": list(getattr(footer, "labels", []) or []),
            "dims": [str(d) for d in stack.dims],
            "shape": [int(v) for v in stack.shape],
            "dtype": str(stack.dtype),
            "compression_type": header.compression_type,
            "compression_level": header.compression_level,
            "description": header.description or None,
            "tag_dictionary": dict(getattr(footer, "tag_dictionary", {}) or {}),
        }

    payload: dict[str, Any] = {
        "source_file": file.name,
        "source_file_id": str(file.id),
        "measurement": group.key,
        "stacks": [
            {
                "channel_index": channel,
                "stack_index": member.index,
                "stack_name": member.name,
                "series_id": text(member.image.id) if member.image is not None else None,
                "obf": obf_record(member.stack),
            }
            for channel, member in enumerate(group.members)
        ],
    }

    images = [m.image for m in group.members if m.image is not None]
    if not images:
        payload["ome"] = None
        return json.dumps(payload, default=str)

    # By id, because every member of an acquisition points at the same instrument and a
    # document listing it four times would not revalidate.
    instruments = list(
        {
            i.id: i for i in (m.instrument for m in group.members) if i is not None
        }.values()
    )

    try:
        from ome_types.model import OME

        reduced = OME(images=images, instruments=instruments)
        payload["ome"] = json.loads(reduced.model_dump_json(exclude_unset=True))
    except Exception:
        # A reduced document can fail to revalidate when an image references something
        # left behind (an experimenter, a plate). The metadata blob is not worth failing
        # an upload over, so fall back to the images alone.
        logger.warning(
            "Could not rebuild a reduced OME document for acquisition %r", group.key
        )
        try:
            payload["ome"] = [
                json.loads(image.model_dump_json(exclude_unset=True))
                for image in images
            ]
        except Exception:
            payload["ome"] = None
    return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Device state
#
# The layer OME records but the light path has no member for: what the software
# wrote down, what the stage was doing, and every device setting that is a number
# on a named box rather than something the light passed through.
# ---------------------------------------------------------------------------


def stack_optikit(
    file: File,
    group: StackGroup,
    scale_note: str | None,
) -> OptikitStateInput | None:
    """The hardware state of the whole acquisition.

    Every OME-derived device degrades to None, so a file with no OME block still carries
    the acquisition record and the OBF stack headers. Read off member 0 apart from those
    headers, which differ per member and get one ``obf-stack-N`` device each.
    """
    index = group.index
    stack = group.stack
    image = group.image
    instrument = group.instrument
    pixels = image.pixels if image is not None else None
    planes = list(pixels.planes) if pixels is not None else []
    first = planes[0] if planes else None
    env = getattr(image, "imaging_environment", None) if image is not None else None
    microscope = getattr(instrument, "microscope", None) if instrument else None
    objective_settings = getattr(image, "objective_settings", None) if image else None
    objective = find_objective(instrument, image)

    stage = None
    if first is not None:
        x = um(first.position_x, first.position_x_unit)
        y = um(first.position_y, first.position_y_unit)
        z = um(first.position_z, first.position_z_unit)
        if any((x, y, z)):
            stage = StageStateInput(x=x, y=y, z=z)

    temperature = None
    if env is not None:
        temperature = qty(env.temperature, env.temperature_unit, "degC")

    state_devices = devices(
        device(
            "microscope",
            "microscope",
            setting("instrument-id", string=text(getattr(instrument, "id", None))),
            setting(
                "manufacturer", string=text(getattr(microscope, "manufacturer", None))
            ),
            setting("model", string=text(getattr(microscope, "model", None))),
            setting(
                "serial-number", string=text(getattr(microscope, "serial_number", None))
            ),
            setting("type", string=text(getattr(microscope, "type", None))),
        ),
        # The run itself. This is the only home the source file's identity has left, so
        # `source-file` and `source-file-id` are load-bearing rather than decorative.
        # Unlike its OME sibling, every setting here that comes from the OME record is
        # guarded — the stack facts above the guard are always available.
        device(
            "acquisition",
            "software",
            setting("source-file", string=file.name),
            setting("source-file-id", string=str(file.id)),
            setting("measurement", string=group.key),
            # What this dataset was merged from. The representative's index and name stay
            # for continuity, but on their own they would name one dye and imply the
            # other three were never here.
            setting("stack-count", number=float(len(group.members))),
            setting(
                "stack-names", string=str([member.name for member in group.members])
            ),
            setting(
                "stack-indices",
                string=str([member.index for member in group.members]),
            ),
            setting("stack-index", number=float(index)),
            setting("stack-name", string=text(stack.name)),
            setting("series-id", string=text(image.id) if image is not None else None),
            setting(
                "image-name", string=text(image.name) if image is not None else None
            ),
            setting(
                "image-description",
                string=text(image.description) if image is not None else None,
            ),
            setting(
                "acquisition-date",
                string=image.acquisition_date.isoformat()
                if image is not None and image.acquisition_date
                else None,
            ),
            setting(
                "creator", string=text(getattr(image, "creator", None)) if image else None
            ),
            setting(
                "dimension-order",
                string=text(pixels.dimension_order) if pixels is not None else None,
            ),
            setting("pixel-type", string=text(pixels.type) if pixels is not None else None),
            setting(
                "significant-bits",
                number=pixels.significant_bits if pixels is not None else None,
            ),
            # Says out loud what the registration below had to assume, and which of the
            # two sources it believed. There is no way to express "unknown" through a
            # BY_DIMENSION edge — omitting an axis and passing 1.0 are the same claim —
            # so the assumption is made visible here rather than hidden in the edge.
            setting("scale-note", string=scale_note),
            setting(
                "time-increment",
                quantity=qty(pixels.time_increment, pixels.time_increment_unit, "s")
                if pixels is not None
                else None,
            ),
        ),
        # One per member: the header is the only part of this state that differs between
        # the stacks a merged dataset was built from, and offsets in particular are here
        # *because* they were refused as a placement.
        *(obf_stack_device(member.stack, member.index) for member in group.members),
        device(
            "objective",
            "objective",
            setting("model", string=text(getattr(objective, "model", None))),
            setting(
                "manufacturer", string=text(getattr(objective, "manufacturer", None))
            ),
            setting(
                "magnification",
                number=getattr(objective, "nominal_magnification", None)
                or getattr(objective, "calibrated_magnification", None),
            ),
            setting("numerical-aperture", number=getattr(objective, "lens_na", None)),
            setting(
                "working-distance",
                quantity=um(
                    getattr(objective, "working_distance", None),
                    getattr(objective, "working_distance_unit", None),
                ),
            ),
            setting("immersion", string=text(getattr(objective, "immersion", None))),
            setting("correction", string=text(getattr(objective, "correction", None))),
            setting(
                "correction-collar",
                number=getattr(objective_settings, "correction_collar", None),
            ),
            setting("medium", string=text(getattr(objective_settings, "medium", None))),
            setting(
                "refractive-index",
                number=getattr(objective_settings, "refractive_index", None),
            ),
        ),
        device(
            "environment",
            "environment",
            setting(
                "air-pressure",
                quantity=qty(
                    getattr(env, "air_pressure", None),
                    getattr(env, "air_pressure_unit", None),
                    "hPa",
                ),
            ),
            setting("humidity", number=getattr(env, "humidity", None)),
            setting("co2-percent", number=getattr(env, "co2_percent", None)),
        ),
        # Only reached when the position could not be read as a length — which is the
        # common case, because OME's schema default for a plane position is "reference
        # frame", an arbitrary unit that is not micrometers. The typed StageStateInput
        # above must not claim it, but the numbers are still what the microscope wrote
        # down, so they are kept verbatim rather than dropped.
        device(
            "stage-xy",
            "stage",
            setting(
                "position-x",
                string=f"{first.position_x} {unit_symbol(first.position_x_unit)}"
                if first is not None
                and first.position_x is not None
                and to_micrometer(first.position_x, first.position_x_unit) is None
                else None,
            ),
            setting(
                "position-y",
                string=f"{first.position_y} {unit_symbol(first.position_y_unit)}"
                if first is not None
                and first.position_y is not None
                and to_micrometer(first.position_y, first.position_y_unit) is None
                else None,
            ),
            setting(
                "position-z",
                string=f"{first.position_z} {unit_symbol(first.position_z_unit)}"
                if first is not None
                and first.position_z is not None
                and to_micrometer(first.position_z, first.position_z_unit) is None
                else None,
            ),
        ),
        device(
            "timing",
            "acquisition",
            setting("plane-count", number=float(len(planes)) if planes else None),
            setting(
                "first-delta-t",
                quantity=qty(first.delta_t, first.delta_t_unit, "s") if first else None,
            ),
            setting(
                "last-delta-t",
                quantity=qty(planes[-1].delta_t, planes[-1].delta_t_unit, "s")
                if planes
                else None,
            ),
            setting(
                "exposure-time",
                quantity=qty(first.exposure_time, first.exposure_time_unit, "s")
                if first
                else None,
            ),
        ),
    )

    if stage is None and temperature is None and not state_devices:
        return None
    return OptikitStateInput(
        stage=stage, temperature=temperature, devices=state_devices
    )


# ---------------------------------------------------------------------------
# The light path
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------


def channel_label(
    channel: Any | None, index: int, implicit_rgb: bool, stack_name: str
) -> str:
    """What to call this channel.

    The one place this diverges from its OME sibling. An OBF file is routinely one
    single-channel stack per dye, with the dye's name on the *stack* —
    ``Overview 8/Image 3/17:28:01/STAR RED`` — so with nothing better the last path
    segment is the label. ``Channel 0`` is not a name a microscopist recognises.

    ``stack_name`` is the source stack's name, not the dataset's: channels are merged, so
    four channels of one dataset have four different source stacks and it is the
    per-stack name that carries the dye.
    """
    if implicit_rgb:
        return (
            ("Red", "Green", "Blue", "Alpha")[index]
            if index < 4
            else f"Sample {index}"
        )
    if channel is not None:
        for candidate in (channel.name, channel.fluor, channel.id):
            rendered = text(candidate)
            if rendered:
                return rendered
    if stack_name:
        return channel_component(stack_name)
    return f"Channel {index}"


def origin_anchor(
    dims: Sequence[str],
    file: File,
    group: StackGroup,
    scale_note: str | None,
) -> CoordinateAnchorInput:
    """Everything true of the whole acquisition, pinned to voxel (0, ..., 0).

    Takes the dim *names* rather than the array, so the caller can drop the reader-backed
    arrays — and the reader's scratch directories — before this is built.

    The label is the acquisition's own name, not any member's: a member's OME image is
    named for its dye, so using it would name a four-channel dataset after one channel.
    """
    return CoordinateAnchorInput(
        axis_anchors=tuple(AxisAnchorInput(axis=str(d), value=0) for d in dims),
        # No label: the acquisition's name is the dataset's name, never a channel
        # label. Channel labels go on the channel anchors, at {c: i}.
        ome_metadata=OmeMetadataInput(metadata_string=stack_metadata_json(file, group)),
        microscope=stack_optikit(file, group, scale_note),
    )


def channel_color(
    channel: Any | None, index: int, implicit_rgb: bool, total: int, offset: int = 0
) -> tuple[int, int, int, int]:
    """What to draw this channel in.

    OME's own colour first — what the acquisition software displayed, and the only one of
    these that is a recorded fact. Then the emission wavelength, which at least lands in
    the right part of the spectrum. Then a palette.

    ``offset`` keeps the palette moving *between* datasets: a merged acquisition walks it
    on its own, but a file holding several would otherwise open each on the same colour,
    and they composite into one scene where that blends to a single wash.
    """
    if implicit_rgb:
        # Index 3 of an RGBA pixel is alpha — a mask, not a colour. `% 3` would paint it
        # the same red as channel 0, which reads as two identical channels in every
        # listing that shows the anchor colour. White is the honest neutral: the layer
        # composite does not use this colour at all (it hardcodes the three primaries),
        # so this only ever labels the anchor.
        if index >= 3:
            return rgba((255, 255, 255))
        return rgba(RGB_PRIMARIES[index])

    if channel is not None:
        from_file = ome_rgba(channel)
        if from_file is not None:
            return from_file

        emission = to_micrometer(
            channel.emission_wavelength, channel.emission_wavelength_unit
        )
        if emission is not None:
            return color_for_wavelength(emission * 1000, COLOR_BY_EMISSION, FAR_RED)

    return fallback_color(index, total, offset)


@dataclass(frozen=True)
class ChannelSpec:
    """One channel, as both the anchors and the layers need it.

    Computed once because the histogram is the expensive part — it is both the layer's
    contrast window and the anchor's render metadata, and computing it twice would double
    the only forced materialisation in the converter.
    """

    index: int
    label: str
    color: tuple[int, int, int, int]
    histogram: ValueHistogramInput
    ome_channel: Any | None
    # The OME image and instrument this channel's optics come from. Carried per channel
    # rather than passed once alongside the specs, because a merged acquisition draws its
    # four channels from four different OME images — one per source stack — and a light
    # path read off the wrong one is wrong in a way nothing downstream can detect.
    ome_image: Any | None
    instrument: Any | None


def channel_specs(
    pyramid: Sequence[xr.DataArray],
    owners: Sequence[tuple[Member, int]],
    implicit_rgb: bool,
    color_offset: int,
) -> list[ChannelSpec]:
    """One spec per channel of the array.

    A contrast window is per channel by nature — a nuclear stain and a brightfield
    channel share no sensible range — so each histogram comes from that channel's slice.

    ``owners`` maps each output channel back to the member it came from and its position
    within it. It is built during the merge because that is the only point where the
    per-member channel counts are still known.

    Reads level 0 off the pyramid rather than the source array: they hold the same values,
    but by then ``build_pyramid`` has copied the lazy base into its own scratch store, so
    the caller can drop the reader's arrays — and its multi-gigabyte scratch directories —
    before any of this is computed.
    """
    base = pyramid[0]
    total = base.sizes["c"]
    specs: list[ChannelSpec] = []

    for index in range(total):
        member, local = owners[index]
        channels = (
            list(member.image.pixels.channels) if member.image is not None else []
        )
        # Implicit RGB has three array channels behind one OME channel, so the metadata
        # does not index by array position and only the first is real.
        channel = channels[local] if local < len(channels) and not implicit_rgb else None

        slice_ = base.isel(c=index)
        if slice_.nbytes > HISTOGRAM_BUDGET_BYTES and len(pyramid) > 1:
            # `from_array` materialises whatever it is handed, so a multi-gigabyte
            # channel is summarised from the coarsest level instead — already built, so
            # it costs nothing. The artifact is worth naming: max downsampling keeps only
            # block maxima, so the distribution shifts *up* and p1/p99 come out brighter
            # than the truth. Hence level 0 wherever it fits.
            slice_ = pyramid[-1].isel(c=index)

        specs.append(
            ChannelSpec(
                index=index,
                label=channel_label(channel, index, implicit_rgb, member.name),
                color=channel_color(channel, index, implicit_rgb, total, color_offset),
                histogram=ValueHistogramInput.from_array(slice_),
                ome_channel=channel,
                ome_image=member.image,
                instrument=member.instrument,
            )
        )

    return specs


def channel_anchors(specs: Sequence[ChannelSpec]) -> list[CoordinateAnchorInput]:
    """One anchor per channel: its contrast window, its identity, its optics."""
    return [
        CoordinateAnchorInput(
            axis_anchors=(AxisAnchorInput(axis="c", value=spec.index),),
            value_histogram=spec.histogram,
            label=LabelInput(label=spec.label),
            microscope=(
                channel_optikit(
                    spec.ome_channel, spec.index, spec.label, spec.instrument
                )
                if spec.ome_channel is not None
                else None
            ),
            light_graph=(
                channel_lightpath(spec.ome_channel, spec.ome_image, spec.instrument)
                if spec.ome_channel is not None
                else None
            ),
        )
        for spec in specs
    ]


def channel_lens(dataset: Any, index: int) -> Any:
    """The lens that pins channel `index`: what a layer over one channel draws.

    A layer draws its lens and nothing narrower, so a channel is a lens of its own.
    A dataset with no channel axis is one signal already, and its whole lens is it.
    """
    if "c" not in dataset.axis_names:
        return dataset.lens()
    return dataset.lens(c=index)


def author_channel_layers(
    mikro: Mikro,
    dataset: ArrayDataset,
    scene: Any,
    specs: Sequence[ChannelSpec],
    is_volume: bool,
    implicit_rgb: bool,
    order_offset: int = 0,
    phasor_axis: str | None = None,
) -> int:
    """One layer per channel of `dataset`, composited into `scene`.

    Per channel, because one layer over a multi-channel dataset means one recipe standing
    in for every channel: no per-channel colour, no contrast, nothing to toggle. It is
    also the only place OME's own channel colour can land.

    The one exception is a packed RGB(A) stack, one layer over its samples. `implicit_rgb`
    is trusted only as far as it can be checked — this function refuses it for any channel
    count that is not 3 or 4, which is the backstop for the whole rule. Kept here rather
    than left to the server because it fires first and demotes instead of failing.

    One lens per channel, beside the dataset's whole one: a layer draws its lens and
    nothing narrower, so the channel a layer shows is the one its lens pins.

    **Which kind of layer** — each recipe has a server kind that states it as fields:

    * one channel, flat → ``create_intensity_layer``
    * one channel, through a z-stack → ``create_volume_layer``
    * a packed RGB(A) pixel, flat → ``create_rgb_layer``

    ``create_layer`` and a render graph are left for the one shape none of them can say,
    a packed RGB(A) **z-stack**: an RGB layer carries no projection mode.

    ``phasor_axis`` overrides all of it — a FLIM arrival-time axis gets one phasor layer
    and no intensity layers at all; see below for why that is an either/or.

    Returns the number of layers created — not ``len(specs)`` in the RGB case.
    """
    lens = dataset.lens()

    if phasor_axis is not None:
        # A lifetime map, and *instead of* the per-channel layers rather than alongside
        # them. An intensity layer indexes the channel axis and says nothing about the
        # arrival-time axis, so over a FLIM cube it would draw one photon-arrival bin --
        # a few counts per pixel, which is shot noise wearing the channel's colour. There
        # is no per-channel layer worth authoring here until something sums that axis.
        #
        # No `createPhasorCalibration`, deliberately. An instrument-response correction has
        # to be measured from a reference acquisition of known lifetime, and nothing in the
        # file this converter is reading is one. An uncalibrated phasor still renders; its
        # hue is a relative lifetime rather than an absolute one, and inventing a
        # correction would make it look like the latter.
        mikro.create_phasor_layer(
            lens=lens.id,
            scene=scene.id,
            phasor_axis=phasor_axis,
            # Harmonic 1: the fundamental, and the only one a single laser period
            # resolves without assuming what the decay looks like.
            harmonic=1,
            transfer=PhasorTransferInput(
                # The phase, which over an arrival-time axis reads as tau_phi.
                mode=PhasorColorMode.PHASE,
                # A pixel with a handful of photons has a phasor, and it is noise. Weighting
                # by intensity is what keeps the background from being a field of confident
                # random lifetimes.
                weight_by_intensity=True,
            ),
            visible=True,
            order=order_offset,
        )
        logger.info(
            "Acquisition has a MICROTIME axis %r; authored a phasor layer over it "
            "instead of %s intensity layer(s).",
            phasor_axis,
            len(specs),
        )
        # This branch preempts the other three, including the packed-sample reading below,
        # so a discarded one is said out loud — the standard the demotion warning sets. A
        # stack claiming both a packed RGB pixel and an arrival-time axis is not something
        # any file here does, and if one turns up the pixel type and the footer's units
        # disagree about what it holds, which is worth seeing.
        if implicit_rgb:
            logger.warning(
                "Acquisition claims both a packed RGB(A) pixel and a MICROTIME axis %r. "
                "Taking the phasor reading: an arrival-time axis is a measured unit in "
                "the footer, where the packed reading is inferred from the pixel type.",
                phasor_axis,
            )
        return 1

    if implicit_rgb and len(specs) not in (3, 4):
        logger.warning(
            "Refusing to composite %s channel(s) into one RGB layer; a render graph "
            "covers exactly one channel unless the channels are the 3 or 4 samples of "
            "one pixel. Authoring one layer per channel instead.",
            len(specs),
        )
        implicit_rgb = False

    if implicit_rgb:
        if is_volume:
            # The one shape the flat kinds cannot say. An RGB layer has no projection
            # mode — its three channels are components of one picture, not three signals
            # to project separately — so a packed RGB z-stack still composites three
            # MIP'd channel nodes by hand. Three primaries even for RGBA: the fourth
            # sample is alpha, which is a mask rather than a colour and has no primary to
            # be painted in.
            #
            # `rgb_graph` says nearly this, but its bare triples are what the server
            # rejects. Same recipe, stated in full.
            mikro.create_layer(
                lens=lens.id,
                scene=scene.id,
                render_graph=composite_graph(
                    [
                        {
                            "lens": channel_lens(dataset, index).id,
                            "color": rgba(primary),
                            "colormap": ColorMap.INTENSITY,
                            # Already display-referred; stretching it would change the
                            # colours the image was authored with.
                            "clim_min": 0.0,
                            "clim_max": 255.0,
                            "mode": ProjectionMode.MIP,
                        }
                        for index, primary in enumerate(RGB_PRIMARIES)
                    ]
                ),
                blending=Blending.ADDITIVE,
                visible=True,
                order=order_offset,
            )
            return 1

        # Flat: the picture is exactly what an RGB layer is for, and it says so as three
        # component lenses, one per sample, rather than as a tree a reader has to
        # recognise. The contrast window is the full 8-bit range for the reason above, and
        # shared across all three — they are one picture, not three signals with their own
        # scaling.
        #
        # No `blending` argument here, and none accepted: the server forces NORMAL,
        # because a photograph sits *over* what is beneath it rather than summing with it.
        # That is a change from the ADDITIVE this path used to pass.
        mikro.create_rgb_layer(
            lens=lens.id,
            scene=scene.id,
            red_lens=channel_lens(dataset, 0).id,
            green_lens=channel_lens(dataset, 1).id,
            blue_lens=channel_lens(dataset, 2).id,
            clim_min=0.0,
            clim_max=255.0,
            visible=True,
            order=order_offset,
        )
        return 1

    for spec in specs:
        # The window the histogram already found. p1/p99 rather than min/max, because a
        # single hot pixel should not decide the display range of the whole channel.
        clim_min = (
            spec.histogram.p1 if spec.histogram.p1 is not None else spec.histogram.min
        )
        clim_max = (
            spec.histogram.p99 if spec.histogram.p99 is not None else spec.histogram.max
        )

        # `color`, not a colormap: what this converter knows about a channel is an RGBA
        # quad — the colour Imspector saved, or the one its emission wavelength implies —
        # and no named map means it. That fact used to force the whole layer onto a render
        # graph; an intensity layer carries it as a field now.
        settings = dict(
            lens=channel_lens(dataset, spec.index).id,
            scene=scene.id,
            color=spec.color,
            clim_min=clim_min,
            clim_max=clim_max,
            blending=Blending.ADDITIVE,
            visible=True,
            order=order_offset + spec.index,
        )

        if is_volume:
            # A z-stack has no single plane to show, so each channel projects through the
            # volume rather than picking one slice. Same kind of layer, one field more.
            mikro.create_volume_layer(mode=ProjectionMode.MIP, **settings)
        else:
            mikro.create_intensity_layer(**settings)

    return len(specs)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


def voxel_scale(
    image: Any | None, stack: Any, dims: Sequence[str]
) -> tuple[dict[str, float] | None, str | None]:
    """The per-axis factor from this dataset's indices into micrometers.

    OME first: it is the calibration Imspector wrote for a reader to use, and it is
    per-image, so it stays right for a derived stack — a maximum projection's
    ``physical_size_z`` is the whole slab, not the spacing it collapsed. The stack header
    is the fallback, and only for axes the footer's SI units call a length.

    Returns ``(None, reason)`` when neither has a lateral calibration. Registering anyway
    is not a neutral default — ``scale={"x": 1.0}`` claims one pixel is one micrometer,
    and the dataset would composite at the wrong size beside calibrated stacks.

    A missing *axial* size is different: omitting ``z`` and passing ``z=1.0`` are the same
    claim, so there is no way to abstain. That assumption goes in the note instead.
    """
    shared = [d for d in dims if d in WORLD_AXES]
    if not shared:
        return None, "no spatial axes to register"

    pixels = image.pixels if image is not None else None
    from_header = header_voxel_sizes(stack)

    sizes: dict[str, float | None] = {}
    sources: dict[str, str] = {}
    for axis in ("x", "y", "z"):
        from_ome = (
            to_micrometer(
                getattr(pixels, f"physical_size_{axis}", None),
                getattr(pixels, f"physical_size_{axis}_unit", None),
            )
            if pixels is not None
            else None
        )
        header_size = from_header.get(axis)

        if from_ome is not None and from_ome > 0:
            sizes[axis], sources[axis] = from_ome, "ome"
        elif header_size is not None and header_size > 0:
            sizes[axis], sources[axis] = header_size, "obf-header"
        else:
            sizes[axis], sources[axis] = None, "unknown"

    if sizes["x"] is None or sizes["y"] is None:
        return (
            None,
            "no lateral pixel size in either the OME record or the stack header — "
            "not registered into a physical space",
        )

    notes = [
        f"{axis} from {sources[axis]}" for axis in shared if sources.get(axis) != "unknown"
    ]
    if "z" in shared and sizes["z"] is None:
        notes.append("physical-size-z unknown, registered as 1 um per plane")

    return (
        {axis: sizes.get(axis) or 1.0 for axis in shared},
        "; ".join(notes) or None,
    )


def plane_positions(image: Any | None) -> dict[int, dict[str, float | None]]:
    """Stage position per index of the OME ``T`` axis, in the world's micrometers.

    Keyed by ``TheT``, first plane found for each: every ``Z`` and ``C`` of one T index
    share a stage position, and only the Z differs, which `voxel_scale` states as a spacing.
    """
    pixels = image.pixels if image is not None else None
    planes = list(pixels.planes) if pixels is not None else []
    out: dict[int, dict[str, float | None]] = {}
    for plane in planes:
        t = int(getattr(plane, "the_t", 0) or 0)
        if t in out:
            continue
        out[t] = {
            "x": to_micrometer(plane.position_x, plane.position_x_unit),
            "y": to_micrometer(plane.position_y, plane.position_y_unit),
            "z": to_micrometer(plane.position_z, plane.position_z_unit),
        }
    return out


def mosaic_tiles(image: Any | None, extent: int) -> dict[int, dict[str, float | None]] | None:
    """The per-tile stage positions if the ``t`` axis is really a position axis.

    Imspector writes an overview *mosaic* as a stack whose tiles are indexed along time,
    and nothing contradicts that reading on its face: the header gives the axis no SI
    unit, and the OME record says ``SizeT=189``. Taken literally that is a 189-frame time
    series at one position, when the truth is 189 tiles over a couple of millimetres.

    What separates them is the stage — a time series revisits one position, a mosaic
    visits a new one each index — so the test is the number of *distinct* ``(x, y)``
    pairs: as many as indices is a mosaic; exactly one is a genuine time series; anything
    between is a multi-position time series, which is real and has no answer here, so it
    is warned about and left alone rather than mis-placed by a guess.

    ``DeltaT`` is deliberately not consulted: it advances across a mosaic too, simply
    because moving the stage takes time. It measures how long the scan took, not what the
    axis is.
    """
    if extent <= 1:
        return None
    positions = plane_positions(image)
    if len(positions) < extent:
        return None

    pairs = {
        (p["x"], p["y"])
        for t, p in positions.items()
        if t < extent and p["x"] is not None and p["y"] is not None
    }
    if len(pairs) <= 1:
        return None
    if len(pairs) < extent:
        logger.warning(
            "The t axis has %s indices but only %s distinct stage positions: this looks "
            "like a time series taken at several positions, which this converter cannot "
            "yet split. Uploading it as a time series at one position.",
            extent,
            len(pairs),
        )
        return None
    return {t: p for t, p in positions.items() if t < extent}


def stage_offsets(
    image: Any | None,
    dims: Sequence[str],
    tile: int | None = None,
) -> dict[str, float]:
    """Where the stage was, in the world's micrometers.

    The same numbers that go into ``StageStateInput`` on the origin anchor, so the state
    and the placement agree by construction. ``tile`` selects one index of the ``t`` axis
    for a mosaic — see `mosaic_tiles`.

    The header's own ``offsets`` are deliberately not a fallback. They are metres and look
    like a position, but every axis of every stack carries the same -0.5 — including the
    channel axis, where a position in metres is meaningless. Reading it literally would
    place every dataset half a metre off-axis. A stack with no OME plane registers at the
    world origin instead.
    """
    positions_by_t = plane_positions(image)
    if not positions_by_t:
        return {}
    # Without a tile this is the acquisition's one position, which for a single-position
    # acquisition is the first plane's — as it always was. With one, it is that tile's
    # own, which is the whole point: a mosaic's tiles differ only here.
    key = 0 if tile is None else tile
    positions = positions_by_t.get(key) or positions_by_t[min(positions_by_t)]
    return {
        axis: positions[axis]
        for axis in dims
        if axis in WORLD_AXES and positions.get(axis) is not None
    }


def compose_scene(
    mikro: Mikro, name: str, coordinate_system: Any, entries: Sequence[Converted]
) -> Scene:
    """A scene over `coordinate_system`, with every layer authored explicitly.

    ``create_scene`` rather than ``stage``: the bootstrap decides how many registered
    sources become layers and which recipe each gets, and collapses a multi-channel
    dataset into one layer. This converter knows what it uploaded.

    ``blending`` is ADDITIVE because that is how fluorescence channels combine.
    ``preferred_view`` is only a viewer hint — the projection that renders a volume lives
    on the layer.
    """
    scene = mikro.create_scene(
        name=name,
        coordinate_system=coordinate_system.id,
        blending=Blending.ADDITIVE,
        preferred_view=(
            PreferredView.THREE_D
            # A phasor entry is excluded even when it has a z axis: `is_volume` asks
            # whether the *data* has depth, and a FLIM z-stack does, but the layer
            # authored over it is a phasor — which carries no projection mode and draws
            # a lifetime map either way. Opening such a scene in 3D advertises a
            # rendering nothing on it can do.
            if any(entry.is_volume and entry.phasor_axis is None for entry in entries)
            else PreferredView.TWO_D
        ),
    )

    # Ordered so a stack's channels stay together in the layer stack rather than
    # interleaving with the next stack's.
    order = 0
    for entry in entries:
        order += author_channel_layers(
            mikro,
            entry.dataset,
            scene,
            entry.specs,
            entry.is_volume,
            entry.implicit_rgb,
            order,
            entry.phasor_axis,
        )

    logger.info(
        "Scene %s over %s: %s layer(s) from %s dataset(s)",
        scene.id,
        scene.world_coordinate_system.name,
        order,
        len(entries),
    )
    return scene


# ---------------------------------------------------------------------------
# One acquisition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Converted:
    """What the action needs to know about an acquisition after it has landed.

    Holds no arrays, which is what lets the action keep one entry per acquisition for the
    whole run without keeping every decompressed stack — and scratch directory — alive.
    """

    dataset: ArrayDataset
    specs: list[ChannelSpec]
    registered: bool
    is_volume: bool
    implicit_rgb: bool
    #: The FLIM arrival-time axis, when the footer's SI units declared one. Carried rather
    #: than re-derived at layer time because the stack it was read from is long gone by
    #: then -- that is the whole point of this dataclass holding no arrays.
    phasor_axis: str | None = None

    @property
    def layer_count(self) -> int:
        """How many layers this acquisition composes into a scene.

        Not ``len(specs)``: a packed RGB(A) stack is one layer over its 3 or 4 samples.
        Kept here so the budget check, the progress line and what `author_channel_layers`
        creates cannot drift. For everything else this is one layer per channel, and that
        is the invariant rather than an implementation detail.
        """
        return 1 if self.implicit_rgb else len(self.specs)


def merge_members(
    group: StackGroup,
    tile: int | None = None,
) -> tuple[xr.DataArray, dict[str, str], list[tuple[Member, int]], bool]:
    """One acquisition's stacks as a single channel-stacked array.

    Returns the merged array, the axis types needing statement, the owner of each output
    channel (see `channel_specs`), and whether the merged ``c`` axis is an RGB(A) sample
    axis rather than a list of channels.

    **That flag is the whole of the RGB decision, and it is a fact rather than a
    heuristic.** True only when the acquisition is a single stack whose pixel type packs
    3 or 4 samples — what `load_stack` reports, and what an RGB photograph is. Two members
    are 2+ channels side by side whatever either packs, so a group of more than one is
    never RGB. No channel count, no OME channel count, and no member count on its own may
    decide otherwise: every guess that has stood here (``c == 3``,
    ``len(ome_channels) == 1``) collapses three dyes or splits a real photograph, and both
    shapes occur in real files.

    The members are concatenated rather than stacked: each already carries a ``c`` axis,
    which is what makes a four-dye acquisition and a single three-sample RGB stack the
    same code path. Every other axis must agree exactly — `group_stacks` grouped on the
    header's shape, so a mismatch means the header and the loaded array disagree, which is
    a genuine inconsistency rather than something to broadcast over.
    """
    arrays: list[xr.DataArray] = []
    extra_types: dict[str, str] = {}
    owners: list[tuple[Member, int]] = []
    packed: list[bool] = []

    for member in group.members:
        array, types, member_packed = load_stack(member.stack, tile=tile)
        packed.append(member_packed)

        if arrays:
            first = arrays[0]
            mismatched = {
                dim: (int(first.sizes[dim]), int(array.sizes.get(dim, -1)))
                for dim in first.dims
                if dim != "c" and int(array.sizes.get(dim, -1)) != int(first.sizes[dim])
            }
            if set(array.dims) != set(first.dims) or mismatched:
                raise ValueError(
                    f"Stack {member.index} ({member.name!r}) does not line up with the "
                    f"rest of {group.key!r}: dims {tuple(array.dims)} vs "
                    f"{tuple(first.dims)}, mismatched {mismatched or '{}'}"
                )
            # Concat takes the first array's dim order for the result, so an aligned but
            # differently-ordered member has to be transposed rather than trusted.
            array = array.transpose(*first.dims)

        extra_types.update(types)
        owners.extend((member, local) for local in range(int(array.sizes["c"])))
        arrays.append(array)

    merged = arrays[0] if len(arrays) == 1 else xr.concat(arrays, dim="c")

    # A packed stack that was grouped with others is not RGB: the samples are three of
    # N channels now. Worth saying out loud rather than silently demoting, because it
    # means the file packed a photograph next to something else.
    rgb_samples = len(arrays) == 1 and packed[0]
    if any(packed) and not rgb_samples:
        logger.info(
            "Acquisition %r merges a packed RGB(A) stack with %s other stack(s); its "
            "samples are ordinary channels here, one layer each.",
            group.key,
            len(arrays) - 1,
        )

    return merged, extra_types, owners, rgb_samples


def scales_agree(
    left: dict[str, float] | None, right: dict[str, float] | None
) -> bool:
    """Whether two calibrations say the same thing, up to float conversion.

    A relative tolerance rather than equality: the sizes are float32 in the record, so the
    sample file's nominal 1 µm reads back as 0.9999999747378753 (2.5e-8 relative). 1e-6
    clears float32 precision, while any disagreement worth warning about is percent-scale.
    """
    if left is None or right is None:
        return left is None and right is None
    if set(left) != set(right):
        return False
    return all(math.isclose(left[axis], right[axis], rel_tol=1e-6) for axis in left)


def check_member_scales(
    group: StackGroup, dims: Sequence[str], chosen: dict[str, float] | None
) -> None:
    """Warn when the members of one acquisition disagree about physical scale.

    The calibration is taken from member 0 and applies to the whole merged dataset. If the
    members disagree, the merge has silently picked one and the rest are drawn at the wrong
    size — nothing downstream can notice, so it is said here. Compared with a tolerance;
    see `scales_agree`.
    """
    for member in group.members[1:]:
        scale, _ = voxel_scale(member.image, member.stack, dims)
        if not scales_agree(scale, chosen):
            logger.warning(
                "Acquisition %r: stack %s (%r) reports scale %s but the dataset is "
                "registered with %s from stack %s",
                group.key,
                member.index,
                member.name,
                scale,
                chosen,
                group.index,
            )


def convert_tile(
    mikro: Mikro,
    group: StackGroup,
    order: int,
    has_ome: bool,
    file: File,
    world: Any,
    levels: int,
    method: str,
    stage_scene: bool,
    tile: int | None = None,
) -> Converted:
    """Upload one dataset — a whole acquisition, or one tile of a mosaic.

    ``order`` is the acquisition's ordinal in the file, used only to keep the fallback
    palette moving between datasets sharing a scene. ``has_ome`` separates "this file
    carries no OME block" from "it does and this acquisition is not in it".
    """
    if group.image is None and has_ome:
        logger.info(
            "Acquisition %r has no matching OME image; converting it without metadata",
            group.key,
        )

    array, extra_types, owners, rgb_samples = merge_members(group, tile=tile)

    # Canonical order — time, then channel, then space — is load-bearing rather than
    # cosmetic: the render axes are derived from the *position* of the spatial axes, so a
    # mis-ordered array does not fail to render, it renders wrongly.
    data = canonical(array, types=extra_types)
    dims = [str(d) for d in data.dims]
    # `extra_types` itself, not `dict.fromkeys(extra_types, "INDEX")`. Every declared axis
    # was an INDEX back when the only thing being declared was "this name means nothing to
    # the convention", so flattening the values was a no-op; a MICROTIME axis makes it a
    # silent downgrade -- to an enumerating axis with no metric, which is the one thing a
    # phasor cannot be taken over. `canonical` on the line above has always read the real
    # values, so the two also disagreed about the same dict.
    axes = axes_for(data, types=extra_types, long_names=AXIS_LONG_NAMES)

    # Read back off the declared types rather than recomputed from the stack: whatever
    # survived the squeeze, the rename and the canonical reordering above is what the
    # dataset will actually carry, and a name derived any earlier can name an axis that
    # is no longer there.
    phasor_axis = next(
        (name for name, kind in extra_types.items() if kind == "MICROTIME" and name in dims),
        None,
    )

    # ONE LAYER PER CHANNEL. The only dataset that is not one layer per channel is a
    # packed RGB(A) photograph, and `merge_members` has already established that from
    # the pixel type — the file's own statement, not an inference from channel counts or
    # from the OME record. Nothing else may narrow this; see `merge_members` for the
    # three heuristics that used to live here and what each one got wrong.
    #
    # The size check is redundant with `load_stack` and kept anyway: it is the single
    # condition under which `author_channel_layers` may emit fewer layers than channels,
    # so it is stated where that consequence is visible.
    implicit_rgb = rgb_samples and data.sizes["c"] in (3, 4)
    if rgb_samples and not implicit_rgb:
        logger.warning(
            "Acquisition %r reports packed samples but carries %s channel(s); treating "
            "them as channels, one layer each.",
            group.key,
            data.sizes["c"],
        )
    is_volume = data.sizes.get("z", 1) > 1

    # build_pyramid rather than dataset_arrays, because the level arrays themselves are
    # needed: the histogram of a very large channel is taken off the coarsest one.
    # scales_from returns levels 1..N only — level 0 *is* `data`, and listing it in both
    # would upload the base twice.
    #
    # Which of two very different costs this is depends on `tile`.
    #
    # Whole-stack (`tile is None`): `asarray` hands back a dask array over a scratch Zarr
    # chunked flat in 1 MB blocks and then reshaped, which is a poor shape to coarsen, so
    # `build_pyramid` materialises it once into a properly blocked store. Every member's
    # reader array is alive at once — concatenation is lazy, and nothing can be released
    # until the pyramid has read through all of it — so this is the peak of the run.
    #
    # Per tile: every member arrived as an eager ndarray of ~16 MB from
    # `ObfStack.read_outer`, so `build_pyramid` takes its non-lazy path and writes no
    # scratch store at all, and the peak is a handful of tiles rather than the file.
    # That is the reason the mosaic path exists at all — see `mosaic_tiles`.
    logger.info(
        "Acquisition %r: building a %s-level pyramid over %s channel(s), %.1f GB",
        group.key,
        levels,
        data.sizes["c"],
        data.nbytes / 1e9,
    )
    pyramid = build_pyramid(data, levels=levels, method=method, types=extra_types)
    scales = scales_from(pyramid, method)

    # Drop the reader's arrays now that the pyramid owns a copy. This matters on the
    # whole-stack path: `ObfStack.asarray` decompresses the stack into a temporary
    # directory that is removed only when the array backing it is collected, so holding
    # these through the upload would mean a second full-size copy on disk — once per
    # member — for the length of the slowest part of the run. On the tile path there is
    # no scratch to release, only ~16 MB per member, but dropping them is still right:
    # the upload below is the long wait, and nothing here is needed across it.
    del array, data

    scale, scale_note = voxel_scale(group.image, group.stack, dims)
    check_member_scales(group, dims, scale)
    specs = channel_specs(pyramid, owners, implicit_rgb, order)

    # A tile's name has to sort: a mosaic arrives as a couple of hundred datasets whose
    # only distinguishing feature is where they sit, and zero-padding is what keeps them
    # in scan order in every listing that sorts by name.
    name = f"{file.name} - {group.key}"
    if tile is not None:
        name = f"{name} - tile {tile:0{len(str(group.stack.outer_size - 1))}d}"
    dataset = mikro.create_array_dataset(
        data=pyramid[0],
        scales=scales,
        name=name,
        axes=axes,
        anchors=[
            origin_anchor(dims, file, group, scale_note),
            *channel_anchors(specs),
        ],
        # Where these arrays came from. Not a coordinate-graph edge — a file has no
        # space, so there is no map to state and `derivedFrom` would be the wrong
        # mechanism — but the link that makes the uploaded dataset traversable back to
        # the bytes it was read out of, and forward from the file to every stack it
        # produced. The stack name identifies the series where the file gives one;
        # `ObfFile.stacks` is keyed by that name, so it is what selected this stack, and
        # the index is the only identifier left when a stack is unnamed.
        #
        # One entry per member, because a merged dataset genuinely has several: each
        # channel is a different series of the same file, and collapsing them to the
        # representative would make three of the four stacks untraceable.
        source_files=[
            SourceFileInput(
                file=file.id,
                series_identifier=(
                    (member.name or str(member.index))
                    if tile is None
                    else f"{member.name or member.index}[{tile}]"
                ),
                # The pixels are the file's pixels: the reader decompresses, this
                # converter renames and reorders axes, and nothing rescales a value. So
                # the histograms and contrast limits here are true of the file too.
                value_relation=ValueRelation.IDENTICAL,
            )
            for member in group.members
        ],
    )

    if scale is None:
        # Uncalibrated: not placed in the shared world, but still renderable in its own
        # pixel grid — which is the honest frame for it. It gets its own scene here
        # rather than waiting for the shared one, because there is no shared space it
        # could be composed into.
        logger.info("Acquisition %r: %s", group.key, scale_note)
        entry = Converted(dataset, specs, False, is_volume, implicit_rgb, phasor_axis)
        if stage_scene:
            if dataset.intrinsic_system is None:
                logger.warning(
                    "Acquisition %r has no intrinsic coordinate system to compose over; "
                    "the dataset is uploaded but has no scene.",
                    group.key,
                )
            else:
                compose_scene(mikro, name, dataset.intrinsic_system, [entry])
        return entry

    world.register(dataset, validity=PlacementValidity.INFERRED, scale=scale, **stage_offsets(group.image, dims, tile))
    return Converted(dataset, specs, True, is_volume, implicit_rgb, phasor_axis)


def convert_group(
    mikro: Mikro,
    group: StackGroup,
    order: int,
    has_ome: bool,
    file: File,
    world: Any,
    levels: int,
    method: str,
    stage_scene: bool,
) -> list[Converted]:
    """Upload one OBF acquisition: one dataset, or one per tile if it is a mosaic.

    An Imspector overview is a *tiled scan*, and the honest shape for that is a dataset
    per tile registered where the stage actually was, rather than one array in which the
    tiles pretend to be a time series. `mosaic_tiles` decides which this is.

    The tiles are read one at a time and each uploaded before the next is read, so what is
    held is a tile, not a stack. Load-bearing: the whole-stack path decompresses ~3 GB per
    member into a scratch directory, which on a tmpfs TMPDIR is 3 GB of RAM per member.
    """
    tiles = (
        mosaic_tiles(group.image, group.stack.outer_size)
        if getattr(group.stack, "supports_outer_reads", False)
        else None
    )
    if tiles is None:
        return [
            convert_tile(
                mikro, group, order, has_ome, file, world, levels, method, stage_scene
            )
        ]

    logger.info(
        "Acquisition %r is a %s-tile mosaic; uploading one dataset per tile",
        group.key,
        len(tiles),
    )
    return [
        convert_tile(
            mikro, group, order, has_ome, file, world, levels, method, stage_scene, tile=tile
        )
        for tile in sorted(tiles)
    ]


# ---------------------------------------------------------------------------
# The action
# ---------------------------------------------------------------------------


@app.action(
    name="Convert OBF file to Datasets",
    description=(
        "Read an Imspector OBF/MSR file and upload one pyramided dataset per "
        "acquisition, merging its single-channel stacks into one multi-channel image "
        "and skipping the projections the software derived for itself. Carries the OME "
        "metadata, light path and device state, placed in a shared micrometer space."
    ),
    collections=["conversion"],
)
def convert_obf_file_to_datasets(
    file: Annotated[
        File,
        Requires(key=EXTENSION, operator=DescriptorOperator.IN, value=["obf", "msr"]),
    ],
    mikro: Mikro,
    task: Task,
    levels: int = 6,
    method: str = "max",
    stage_scene: bool = True,
    world_name: str = "",
) -> list[ArrayDataset]:
    """Convert every acquisition in an OBF file into one multi-channel dataset.

    The unit is an acquisition, not a stack: Imspector writes each dye as its own stack
    and its own maximum projections back into the file, so a four-channel z-stack arrives
    as eight. Derived stacks are skipped and the rest merged by acquisition — see
    `group_stacks`.

    Returns a list rather than yielding. The scene has to be composed after every
    registration, and a generator's post-loop code only runs if the consumer exhausts it —
    a `finally` would fire during garbage collection of an abandoned generator, issuing a
    network call from a context where the transport may be gone, and compose a silently
    partial scene.
    """
    assert file.store, "No file provided"

    task.progress(0, "Downloading file")
    # `FileTrait.as_path`, which replaced thirty lines of `mkdtemp` + `store.download` +
    # a `finally` here. It keeps every property those had — a scratch directory of its
    # own under MIKRO_SCRATCH_DIR else TMPDIR, the local file named after `file.name`
    # rather than the opaque store key, and the directory removed however the block
    # leaves, a part-way download included. Which matters because `store.download(name)`
    # writes relative to the *working* directory, and the cleanup then deleted what it
    # found there: it destroyed `test.ims` in this directory on 2026-08-19.
    #
    # Still the caller's call: nothing on either side knows the file's size, and where
    # /tmp is a tmpfs the default is RAM. Set MIKRO_SCRATCH_DIR for a large file.
    #
    # First converter to make the move — imaris, metamorph, ome and nifti still carry
    # the hand-rolled block.
    with file.as_path() as path:
        task.progress(5, "Reading the OBF header")
        # squeeze=False so length-1 dimensions survive into `sizes`. Two things depend on
        # it: `stack_axis_indices` can only line the array up against the header's
        # per-dimension geometry while the two have the same rank, and `load_stack` needs
        # to make the keep-or-drop decision itself rather than inherit the reader's.
        with ObfFile(path, squeeze=False) as obf:
            ome = try_parse_ome(obf)
            if ome is None:
                logger.info("No OME metadata in %s; converting geometry only", file.name)

            world = space_3d(
                mikro,
                world_name or f"{file.name} (micrometers)", unit=Unit("micrometer")
            )

            stacks = list(obf.stacks)
            # `ObfFile.stacks` is keyed by stack name, so two stacks sharing one name
            # collapse into a single entry. Logged rather than passed over: a silent
            # count is indistinguishable from full coverage.
            logger.info("Read %s stack(s) from %s", len(stacks), file.name)
            groups = group_stacks(stacks, ome)

            converted: list[Converted] = []
            failures: list[tuple[str, Exception]] = []

            for order, group in enumerate(groups):
                percent = 10 + int(85 * order / max(len(groups), 1))
                task.progress(
                    percent,
                    f"Converting acquisition {order + 1}/{len(groups)}: {group.key} "
                    f"({len(group.members)} channels)",
                )
                try:
                    # extend, not append: a mosaic acquisition lands as one dataset per
                    # tile, and every one of them is a dataset this action returns.
                    converted.extend(
                        convert_group(
                            mikro,
                            group,
                            order,
                            ome is not None,
                            file,
                            world,
                            levels,
                            method,
                            stage_scene,
                        )
                    )
                except Exception as exc:  # one bad acquisition must not cost the rest
                    logger.exception("Acquisition %r failed", group.key)
                    failures.append((group.key, exc))
                    continue

            datasets = [c.dataset for c in converted]
            registered = [c for c in converted if c.registered]

            if failures and not datasets:
                raise RuntimeError(
                    f"All {len(failures)} acquisitions failed; first: {failures[0][1]}"
                ) from failures[0][1]
            if failures:
                logger.warning(
                    "%s of %s acquisitions failed: %s",
                    len(failures),
                    len(groups),
                    [key for key, _ in failures],
                )

            # One scene over the shared world, after every registration exists. A layer
            # reaches the world through the registration edge its dataset already has, so
            # composing earlier would simply miss whatever landed later.
            if stage_scene and registered:
                # Not the number of datasets: a 16-channel acquisition is 16 layers on
                # its own, and an RGB stack is the other direction — three samples, one
                # layer. A mosaic multiplies both by its tile count.
                layer_count = sum(c.layer_count for c in registered)
                if layer_count > MAX_SCENE_LAYERS:
                    logger.warning(
                        "%s dataset(s) would compose %s layers; not building a scene "
                        "over more than %s. The world holds every registration and "
                        "outlives this run, so a subset can be composed later. A mosaic "
                        "reaches this on its own: one dataset per tile, one layer per "
                        "channel of each.",
                        len(registered),
                        layer_count,
                        MAX_SCENE_LAYERS,
                    )
                else:
                    task.progress(97, f"Composing {layer_count} layers into one scene")
                    compose_scene(mikro, file.name, world, registered)

            task.progress(100, f"Converted {len(groups)} acquisition(s) into {len(datasets)} dataset(s)")
            return datasets


if __name__ == "__main__":
    run(app)
