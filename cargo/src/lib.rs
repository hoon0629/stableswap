//! 2-coin StableSwap implementation (integer math, 6-decimals).
//!
//! Invariant (n=2, n^n=4):
//!   A * 4 * (x0 + x1) + D = A * D * 4 + D^3 / (4 * x0 * x1)
//!
//! Public API matches the requested deliverable. Internals use u128 with
//! floor-division and conservative rounding to mirror on-chain behavior.

use core::cmp::min;

/// 6-decimal base units (e.g., 1 USDC = 1_000_000).
pub const SCALE_6: u64 = 1_000_000;

// Constants for n = 2
const N_COINS: u128 = 2;
const N_POW_N: u128 = 4;
const MAX_ITERS: usize = 255;

/// Errors for quoting and math.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwapError {
    InvalidIndex,
    ZeroAmount,
    InsufficientLiquidity,
    MathOverflow,
    ConvergenceFailure,
    CastOverflow,
}

/// Deliverable struct
pub struct StableSwapPool {
    pub reserves: [u64; 2],
    pub amplification_coefficient: u64,
    pub fee_bps: u16,
}

impl StableSwapPool {
    /// Create a new pool (fee defaults to 0 bps).
    pub fn new(reserves: [u64; 2], amp: u64) -> Self {
        Self {
            reserves,
            amplification_coefficient: amp,
            fee_bps: 0,
        }
    }

    /// Quote output token `j` for input `dx` of token `i` (post-fee).
    ///
    /// Steps: compute D; add dx to i; solve y (new j balance) s.t. D unchanged;
    /// dy = old_j - y; conservative rounding (−1); apply fee; bounds checks.
    pub fn get_dy(&self, i: usize, j: usize, dx: u64) -> Result<u64, SwapError> {
        if i >= 2 || j >= 2 || i == j {
            return Err(SwapError::InvalidIndex);
        }
        if dx == 0 {
            return Err(SwapError::ZeroAmount);
        }
        let [x0, x1] = self.reserves;
        if x0 == 0 || x1 == 0 {
            return Err(SwapError::InsufficientLiquidity);
        }

        let mut xp = [x0 as u128, x1 as u128];
        let dx_u128 = dx as u128;

        let d0 = self.compute_d_u128()?; // Result<u128, SwapError>

        xp[i] = xp[i].checked_add(dx_u128).ok_or(SwapError::MathOverflow)?;

        let y_new = self.solve_y(&xp, d0, j)?;
        let mut dy = xp[j].checked_sub(y_new).ok_or(SwapError::MathOverflow)?;
        // Conservative rounding (avoid over-credit due to flooring):
        dy = dy.saturating_sub(1);

        // Output fee on dy: dy_net = dy * (10000 - fee) / 10000
        let dy_net = mul_div(dy, 10_000u128 - self.fee_bps as u128, 10_000u128)?;

        // Leave at least 1 base unit of token j in the pool
        let reserve_j = self.reserves[j] as u128;
        if reserve_j == 0 || dy_net >= reserve_j.saturating_sub(1) {
            return Err(SwapError::InsufficientLiquidity);
        }

        u128_to_u64(dy_net)
    }

    /// Compute invariant D (floored to u64). Returns 0 on failure.
    pub fn get_d(&self) -> u64 {
        match self.compute_d_u128() {
            Ok(v) => u128_to_u64(v).unwrap_or(u64::MAX),
            Err(_) => 0,
        }
    }

    /// Compare StableSwap vs constant-product (xy=k) for `amount` from token 0→1.
    /// Returns **improvement** in bps = 10_000 * (dy_ss - dy_xyk) / dy_xyk (floored).
    /// If SS is equal/worse or any path fails, returns 0.
    pub fn calculate_slippage_bps(&self, amount: u64) -> u16 {
        if amount == 0 || self.reserves[0] == 0 || self.reserves[1] == 0 {
            return 0;
        }
        let dy_ss = match self.get_dy(0, 1, amount) {
            Ok(v) => v as u128,
            Err(_) => return 0,
        };
        if dy_ss == 0 {
            return 0;
        }

        // Constant-product with the SAME rounding + fee policy (output fee).
        let dy_xyk = match self.constant_product_quote(0, 1, amount) {
            Ok(v) => v as u128,
            Err(_) => return 0,
        };
        if dy_xyk == 0 || dy_ss <= dy_xyk {
            return 0;
        }
        let num = (dy_ss - dy_xyk).saturating_mul(10_000u128);
        let bps = num / dy_xyk;
        min(bps as u64, u16::MAX as u64) as u16
    }

    // ----------------
    // Internal math
    // ----------------

    /// Curve-style iterative solver for D (u128).
    fn compute_d_u128(&self) -> Result<u128, SwapError> {
        let x0 = self.reserves[0] as u128;
        let x1 = self.reserves[1] as u128;
        let s = x0.checked_add(x1).ok_or(SwapError::MathOverflow)?;
        if s == 0 {
            return Ok(0);
        }

        let a = self.amplification_coefficient as u128;
        let ann = a.checked_mul(N_POW_N).ok_or(SwapError::MathOverflow)?;

        let mut d = s;
        for _ in 0..MAX_ITERS {
            // D_P = D * D/(x0*2) * D/(x1*2)
            let mut d_p = d;
            d_p = mul_div(d_p, d, x0.checked_mul(N_COINS).ok_or(SwapError::MathOverflow)?)?;
            d_p = mul_div(d_p, d, x1.checked_mul(N_COINS).ok_or(SwapError::MathOverflow)?)?;

            // D_next = ((Ann*S + D_P*2) * D) / ((Ann-1)*D + (3)*D_P)
            let term1 = ann.checked_mul(s).ok_or(SwapError::MathOverflow)?;
            let term2 = d_p.checked_mul(N_COINS).ok_or(SwapError::MathOverflow)?;
            let sum = term1.checked_add(term2).ok_or(SwapError::MathOverflow)?;
            let numerator = sum.checked_mul(d).ok_or(SwapError::MathOverflow)?;

            let den_left = ann
                .checked_sub(1)
                .ok_or(SwapError::MathOverflow)?
                .checked_mul(d)
                .ok_or(SwapError::MathOverflow)?;
            let den_right = (N_COINS + 1)
                .checked_mul(d_p)
                .ok_or(SwapError::MathOverflow)?;
            let denominator = den_left.checked_add(den_right).ok_or(SwapError::MathOverflow)?;
            if denominator == 0 {
                return Err(SwapError::MathOverflow);
            }

            let d_next = numerator / denominator;
            let diff = if d_next > d { d_next - d } else { d - d_next };
            d = d_next;
            if diff <= 1 {
                return Ok(d);
            }
        }

        Err(SwapError::ConvergenceFailure)
    }

    /// Solve for new y (balance of coin j) keeping D constant, after xp[i] increased.
    fn solve_y(&self, xp: &[u128; 2], d: u128, j: usize) -> Result<u128, SwapError> {
        if j >= 2 {
            return Err(SwapError::InvalidIndex);
        }
        let a = self.amplification_coefficient as u128;
        let ann = a.checked_mul(N_POW_N).ok_or(SwapError::MathOverflow)?;

        // S_ = sum of balances excluding j (only one "other" in 2-coin)
        let s_ = xp[1 - j];

        // c = D * D/(x_other*2) * D/(Ann*2)
        let mut c = d;
        c = mul_div(
            c,
            d,
            xp[1 - j]
                .checked_mul(N_COINS)
                .ok_or(SwapError::MathOverflow)?,
        )?;
        c = mul_div(
            c,
            d,
            ann.checked_mul(N_COINS)
                .ok_or(SwapError::MathOverflow)?,
        )?;

        // b = S_ + D/Ann
        let b = s_.checked_add(d / ann).ok_or(SwapError::MathOverflow)?;

        // Iterate: y = (y^2 + c) / (2y + b - D)
        let mut y = d;
        for _ in 0..MAX_ITERS {
            let y_prev = y;

            let num = y
                .checked_mul(y)
                .ok_or(SwapError::MathOverflow)?
                .checked_add(c)
                .ok_or(SwapError::MathOverflow)?;
            let den = y
                .checked_mul(2)
                .ok_or(SwapError::MathOverflow)?
                .checked_add(b)
                .ok_or(SwapError::MathOverflow)?
                .checked_sub(d)
                .ok_or(SwapError::MathOverflow)?;

            if den == 0 {
                return Err(SwapError::MathOverflow);
            }
            y = num / den;

            let diff = if y > y_prev { y - y_prev } else { y_prev - y };
            if diff <= 1 {
                return Ok(y);
            }
        }
        Err(SwapError::ConvergenceFailure)
    }

    /// Constant-product (xy=k) quote with the SAME rounding + fee policy (output fee).
    fn constant_product_quote(&self, i: usize, j: usize, dx: u64) -> Result<u64, SwapError> {
        if i >= 2 || j >= 2 || i == j {
            return Err(SwapError::InvalidIndex);
        }
        if dx == 0 {
            return Err(SwapError::ZeroAmount);
        }
        let x = self.reserves[i] as u128;
        let y = self.reserves[j] as u128;
        if x == 0 || y == 0 {
            return Err(SwapError::InsufficientLiquidity);
        }

        // no input fee; add dx to x, compute out, subtract 1, output-fee
        let x_new = x.checked_add(dx as u128).ok_or(SwapError::MathOverflow)?;
        if x_new == 0 {
            return Err(SwapError::MathOverflow);
        }
        let k = x.checked_mul(y).ok_or(SwapError::MathOverflow)?;
        let y_new = k / x_new;
        let mut out = y.checked_sub(y_new).ok_or(SwapError::MathOverflow)?;
        out = out.saturating_sub(1);

        let out_net = mul_div(out, 10_000u128 - self.fee_bps as u128, 10_000u128)?;
        u128_to_u64(out_net)
    }
}

// --------- helpers ---------

#[inline]
fn mul_div(a: u128, b: u128, denom: u128) -> Result<u128, SwapError> {
    let prod = a.checked_mul(b).ok_or(SwapError::MathOverflow)?;
    if denom == 0 {
        return Err(SwapError::MathOverflow);
    }
    Ok(prod / denom) // floor
}

#[inline]
fn u128_to_u64(x: u128) -> Result<u64, SwapError> {
    if x > u64::MAX as u128 {
        return Err(SwapError::CastOverflow);
    }
    Ok(x as u64)
}

// ---------------- Tests ----------------

#[cfg(test)]
mod tests {
    use super::*;

    fn pool_balanced(amp: u64, fee_bps: u16) -> StableSwapPool {
        let x = 10_000u64 * SCALE_6;
        let mut p = StableSwapPool::new([x, x], amp);
        p.fee_bps = fee_bps;
        p
    }

    #[test]
    fn d_near_sum_when_balanced() {
        let p = pool_balanced(100, 0);
        let d = p.get_d();
        let s = p.reserves[0] + p.reserves[1];
        assert!(d <= s && s - d <= 1, "D={} vs S={}", d, s);
    }

    #[test]
    fn swap_preserves_d_approximately() {
        // fee = 0, so dy_net ≈ raw dy (minus conservative 1)
        let p = pool_balanced(200, 0);
        let d0 = p.get_d();

        let dx = SCALE_6 / 2; // 0.5 unit
        let dy = p.get_dy(0, 1, dx).expect("quote");
        assert!(dy > 0);

        // Post-swap reserves (using dy_net since fee=0 here)
        let post = StableSwapPool::new(
            [p.reserves[0] + dx, p.reserves[1] - dy],
            p.amplification_coefficient,
        );
        let d1 = post.get_d();
        let diff = if d1 > d0 { d1 - d0 } else { d0 - d1 };
        assert!(
            diff <= 2,
            "D drift too large: d0={}, d1={}, diff={}",
            d0,
            d1,
            diff
        );
    }

    #[test]
    fn stableswap_vs_xyk_improvement_small_trade() {
        // Both paths use identical rounding + output fee
        let p = pool_balanced(300, 30);
        let amount = SCALE_6 / 10; // 0.1 unit
        let dy_ss = p.get_dy(0, 1, amount).unwrap();
        let dy_xyk = p.constant_product_quote(0, 1, amount).unwrap();
        assert!(
            dy_ss >= dy_xyk,
            "StableSwap should be >= constant product near peg: ss={}, xyk={}",
            dy_ss,
            dy_xyk
        );
        let _bps = p.calculate_slippage_bps(amount); // sanity: executes without panic
    }

    #[test]
    fn errors_and_guards() {
        let mut p = pool_balanced(100, 0);
        assert_eq!(p.get_dy(0, 0, 1).unwrap_err(), SwapError::InvalidIndex);
        assert_eq!(p.get_dy(2, 1, 1).unwrap_err(), SwapError::InvalidIndex);
        assert_eq!(p.get_dy(0, 1, 0).unwrap_err(), SwapError::ZeroAmount);

        // Tiny reserves -> large dx should fail InsufficientLiquidity (dust guard)
        p.reserves = [1_000, 1_000];
        assert_eq!(
            p.get_dy(0, 1, 10_000).unwrap_err(),
            SwapError::InsufficientLiquidity
        );
    }
}
