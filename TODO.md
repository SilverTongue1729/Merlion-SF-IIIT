# TODO

## Completed ✓
- ✅ **SpikeAmplification Transform** - Custom data preprocessing transform for improving high quantile loss performance
  - Implementation: `merlion/transform/normalize.py`
  - Tests: `tests/transform/test_spike_amplification.py` (6/6 passing)
  - Integration: Registered in `TransformFactory`
  - Documentation: `docs/SpikeAmplification_README.md`, `docs/SpikeAmplification_QuickRef.md`
  - Example: `examples/transform_spike_amplification_example.py`
  - Status: Fully functional and tested

## In Progress / To Do
- [ ] Quantile Loss metric
- [ ] Integrating MOIRAI forecasting model
- [ ] Add other data augmentations we've discussed
- [ ] Look into other ensemble test time augmentation methods

## Notes
See `SPIKE_AMPLIFICATION_SUMMARY.md` for complete details on the SpikeAmplification implementation.
