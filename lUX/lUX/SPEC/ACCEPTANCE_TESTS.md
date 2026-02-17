# Acceptance Tests

- /gallery renders fixture suite without console errors.
- Mode dial changes only center content; outer frame bounding boxes are stable.
- DataValue rendering: missing => 'Data Unavailable' chip; invalid => 'Invalid' chip.
- Tampered fixture must display 'BROKEN_CHAIN' verification state.
- Token purity: no hardcoded colors in production source.
