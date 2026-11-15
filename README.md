# objreader

<!-- arkitekt:badge -->
[![Open in Arkitekt](https://arkitekt.live/img/badge/open-in-arkitekt.svg)](https://arkitekt.live/deeplink?orkestrator=%2Fdo%2Fv1%2Frepo.install%3Fsource%3Doci%253Aghcr.io%252Farkitektio%252Fobjreader)
<!-- /arkitekt:badge -->

An Arkitekt app that converts Imspector OBF/MSR files into mikro datasets.

Its one action, `convert_obf_file_to_datasets`, takes a mikro `File` with an `.obf` or
`.msr` extension and uploads one pyramided, multi-channel `ArrayDataset` per
acquisition. Imspector writes one single-channel stack per dye and writes its own
maximum projections back into the file; the projections are skipped and the remaining
stacks are merged along the channel axis. The OME metadata, light path and device state
travel with each dataset, and every calibrated acquisition is placed in one shared
micrometer space with a scene over it.

The reader is the package's own fork of `obffile` (`src/objreader/obffile.py`), which
streams a compressed stack into a scratch Zarr instead of decompressing it into memory.
Set `MIKRO_SCRATCH_DIR` for large files where `/tmp` is a tmpfs.

```
uv run arkitekt run dev
```
