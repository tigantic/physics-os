# Glossary

- **TT**: Tensor Train decomposition.
- **QTT**: Quantized Tensor Train; reshaping via binary factorization (2×2×…×2) then TT.
- **MPS**: Matrix Product State (TT for vectors/tensors).
- **MPO**: Matrix Product Operator (TT for linear operators).
- **Rank (TT-rank)**: internal bond dimensions; governs compute/memory.
- **Rounding**: truncation/compression step to control rank growth.
- **TT-SVD**: SVD-based TT decomposition.
- **TT-Cross / TCI**: cross interpolation construction using adaptive sampling.
- **Never Go Dense**: product rule; avoid materializing dense arrays.
