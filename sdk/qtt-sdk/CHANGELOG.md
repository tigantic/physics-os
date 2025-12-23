# Changelog

All notable changes to QTT-SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-22

### Added

- Initial release of QTT-SDK
- Core QTTState and MPO data structures
- Conversion functions: `dense_to_qtt`, `qtt_to_dense`
- Arithmetic operations: `qtt_add`, `qtt_scale`, `qtt_subtract`
- Inner products and norms: `qtt_inner_product`, `qtt_norm`
- Element-wise operations: `qtt_elementwise_product`
- Truncation/recompression: `truncate_qtt`
- MPO operators: `identity_mpo`, `shift_mpo`, `derivative_mpo`, `laplacian_mpo`
- MPO application: `apply_mpo`
- Comprehensive test suite
- Big Data analytics example
- Digital Twin example
- Full API documentation

### Performance

- Verified compression ratios up to 83,000x for smooth functions
- Tested on grids up to 2^30 (1 billion) points
- All operations complete in milliseconds at billion-point scale

### Documentation

- Complete API reference in README
- Theory section explaining tensor-train decomposition
- Use case guidelines for when QTT is appropriate
- Benchmark results table
