# stableswap
StableSwap (USDC/USDT) — Rust, 2-coin, 6-decimals

A minimal, production-style implementation of Curve’s StableSwap invariant for a USDC/USDT pair. It uses integer math (u128 intermediates, 6-decimal base units), Newton/fixed-point solvers for the invariant and swap math, and includes unit tests, edge-case guards, and an apples-to-apples slippage comparison vs constant-product (xy=k).