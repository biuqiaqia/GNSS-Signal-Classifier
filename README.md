# GNSS Signal Classifier

Research code for classifying urban GNSS measurements into signal-propagation categories such as line-of-sight, reflection, diffraction, and multipath.

## Workflow

1. `DataProcess/` converts RINEX observations and broadcast ephemerides into per-satellite features.
2. `DataTag/` uses building models, sky masks, and ray tracing to generate signal-type labels.
3. `models/` contains the later RF, BiLSTM, ConvLSTM, and masked-ConvLSTM experiments.
4. `Visualized_Path/` contains utilities for exporting trajectories to KML.

## Notes

- The repository contains research code and selected model artifacts, not the full raw-data archive.
- Paths in older MATLAB scripts may need to be changed for a new machine.
- Bundled third-party MATLAB libraries retain their original licenses.

