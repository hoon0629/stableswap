# stableswap
StableSwap (USDC/USDT) — Rust, 2-coin, 6-decimals

A minimal implementation of Curve’s StableSwap invariant for a USDC/USDT pair. It uses Newton's method for the invariant and swap math, and includes unit tests, edge-case guards, and slippage comparison vs constant-product (xy=k).


This codebase implements a two-asset StableSwap AMM for a USDC/USDT pool using 6-decimal integer math. In lib.rs, StableSwapPool quotes swaps by solving Curve’s invariant with Newton's methods (compute D, then solve the opposing balance to keep D constant), applies conservative rounding and an output-side fee, and exposes helpers for USDC→USDT/USDT→USDC, spot price, and a fair constant-product (xy=k) comparator. It includes apply_swap, plus robust guards for invalid indices, zero amounts, empty reserves, overflow, non-convergence, and dust-drain protection. In main.rs, the default demo prints D, spot price, and sample quotes with an “improvement vs xy=k” metric.