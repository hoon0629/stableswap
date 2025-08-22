//! 2-coin StableSwap (USDC/USDT), integer math, 6-decimals.
//!
//! Indexing & notation (fixed in this implementation):
//!   x0 = reserves[USDC]  (USDC balance)
//!   x1 = reserves[USDT]  (USDT balance)
//!
//! Invariant (n=2, n^n=4):
//!   A * 4 * (x0 + x1) + D = A * D * 4 + D^3 / (4 * x0 * x1)
//!
//! - Reserves & amounts are 6-decimal base units (1 token = 1_000_000).
//! - All core math uses u128; divisions floor; conservative rounding (−1).
//! - Fees are applied on OUTPUT (bps), after conservative rounding.
//! - Dust guard leaves at least 1 base unit on the output side.

use core::cmp::min;

/// 6-decimal base units (1 token = 1_000_000).
pub const SCALE_6: u64 = 1_000_000;

/// Token index aliases (USDC = 0, USDT = 1)
pub const USDC: usize = 0;
pub const USDT: usize = 1;

// Constants for n = 2
const N_COINS: u128 = 2;
const N_POW_N: u128 = 4;
const MAX_ITERS: usize = 1000;

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

/// StableSwap pool for a USDC/USDT pair.
pub struct StableSwapPool {
    /// Reserves in base units: [USDC, USDT]
    pub reserves: [u64; 2],
    /// Amplification coefficient A
    pub amplification_coefficient: u64,
    /// Fee in basis points (e.g., 30 = 0.30%)
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

    /// Convenience constructor for USDC/USDT with explicit fee.
    pub fn new_usdc_usdt(usdc_reserve: u64, usdt_reserve: u64, amp: u64, fee_bps: u16) -> Self {
        Self {
            reserves: [usdc_reserve, usdt_reserve],
            amplification_coefficient: amp,
            fee_bps,
        }
    }

    /// Quote output token `j` for input `dx` of token `i` (post-fee).
    ///
    /// Flow: compute D -> add dx to i -> solve y for j keeping D constant ->
    /// raw dy = old_j - y -> conservative rounding (−1) -> apply output fee ->
    /// dust/liquidity guard -> return dy_net.
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

        let d0 = self.compute_d_u128()?; // D at current state

        xp[i] = xp[i].checked_add(dx_u128).ok_or(SwapError::MathOverflow)?;

        let y_new = self.solve_y(&xp, d0, j)?;
        let mut dy = xp[j].checked_sub(y_new).ok_or(SwapError::MathOverflow)?;
        // Conservative rounding (avoid over-credit under floor division)
        dy = dy.saturating_sub(1);

        // Output fee: dy_net = dy * (10000 - fee) / 10000
        let dy_net = mul_div(dy, 10_000u128 - self.fee_bps as u128, 10_000u128)?;

        // Dust/liquidity guard: leave >= 1 base unit of token j
        let reserve_j = self.reserves[j] as u128;
        if reserve_j == 0 || dy_net >= reserve_j.saturating_sub(1) {
            return Err(SwapError::InsufficientLiquidity);
        }

        u128_to_u64(dy_net)
    }

    /// Execute a swap and **mutate reserves**.
    /// Returns dy (post-fee) actually sent out.
    pub fn apply_swap(&mut self, i: usize, j: usize, dx: u64) -> Result<u64, SwapError> {
        let dy = self.get_dy(i, j, dx)?;
        // Update pool state: add input, remove output
        self.reserves[i] = self
            .reserves[i]
            .checked_add(dx)
            .ok_or(SwapError::MathOverflow)?;
        self.reserves[j] = self
            .reserves[j]
            .checked_sub(dy)
            .ok_or(SwapError::MathOverflow)?;
        Ok(dy)
    }

    /// USDC -> USDT quote (post-fee).
    pub fn get_dy_usdc_to_usdt(&self, dx_usdc: u64) -> Result<u64, SwapError> {
        self.get_dy(USDC, USDT, dx_usdc)
    }

    /// USDT -> USDC quote (post-fee).
    pub fn get_dy_usdt_to_usdc(&self, dx_usdt: u64) -> Result<u64, SwapError> {
        self.get_dy(USDT, USDC, dx_usdt)
    }

    /// Compute invariant D (floored to u64). Returns 0 on failure.
    pub fn get_d(&self) -> u64 {
        match self.compute_d_u128() {
            Ok(v) => u128_to_u64(v).unwrap_or(u64::MAX),
            Err(_) => 0,
        }
    }

    /// Integer (floored) improvement vs constant-product for a USDC→USDT trade.
    /// Returns 0 if SS ≤ XY or on any error.
    pub fn calculate_slippage_bps(&self, amount: u64) -> u16 {
        if amount == 0 || self.reserves[USDC] == 0 || self.reserves[USDT] == 0 {
            return 0;
        }
        let dy_ss = match self.get_dy(USDC, USDT, amount) {
            Ok(v) => v as u128,
            Err(_) => return 0,
        };
        let dy_xyk = match self.constant_product_quote(USDC, USDT, amount) {
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

    /// Floating-point improvement vs constant-product (can be < 1 bps). 0.0 on error.
    pub fn calculate_slippage_bps_f64(&self, amount: u64) -> f64 {
        if amount == 0 || self.reserves[USDC] == 0 || self.reserves[USDT] == 0 {
            return 0.0;
        }
        let dy_ss = match self.get_dy(USDC, USDT, amount) {
            Ok(v) => v as f64,
            Err(_) => return 0.0,
        };
        let dy_xyk = match self.constant_product_quote(USDC, USDT, amount) {
            Ok(v) => v as f64,
            Err(_) => return 0.0,
        };
        if dy_xyk <= 0.0 || dy_ss <= dy_xyk {
            return 0.0;
        }
        10_000.0 * (dy_ss - dy_xyk) / dy_xyk
    }

    /// Spot price (USDT per 1 USDC) at the current state using the 2-coin constant-D differential.
    /// p = (∂F/∂x0)/(∂F/∂x1) = [4A + D^3/(4 x0^2 x1)] / [4A + D^3/(4 x0 x1^2)]
    pub fn spot_price_usdc_in_usdt(&self) -> f64 {
        let x0_usdc = self.reserves[USDC] as f64;
        let x1_usdt = self.reserves[USDT] as f64;
        if x0_usdc <= 0.0 || x1_usdt <= 0.0 {
            return f64::NAN;
        }
        let d = self.get_d() as f64;
        let a = self.amplification_coefficient as f64;
        let d3 = d * d * d;
        let num = 4.0 * a + d3 / (4.0 * x0_usdc * x0_usdc * x1_usdt);
        let den = 4.0 * a + d3 / (4.0 * x0_usdc * x1_usdt * x1_usdt);
        num / den
    }

    /// Spot price (USDC per 1 USDT), reciprocal of `spot_price_usdc_in_usdt`.
    pub fn spot_price_usdt_in_usdc(&self) -> f64 {
        let p = self.spot_price_usdc_in_usdt();
        if !p.is_finite() || p == 0.0 {
            f64::NAN
        } else {
            1.0 / p
        }
    }

    // ----------------
    // Internal math
    // ----------------

    /// Curve-style iterative solver for D (u128).
    fn compute_d_u128(&self) -> Result<u128, SwapError> {
        let x0 = self.reserves[USDC] as u128;
        let x1 = self.reserves[USDT] as u128;
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

            // D_next = ((Ann*S + D_P*2) * D) / ((Ann-1)*D + 3*D_P)
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

    /// Constant-product (xy=k) comparator with the SAME rounding + output-fee policy.
    /// Directional: x_in = reserves[i], y_out = reserves[j].
    fn constant_product_quote(&self, i: usize, j: usize, dx: u64) -> Result<u64, SwapError> {
        if i >= 2 || j >= 2 || i == j {
            return Err(SwapError::InvalidIndex);
        }
        if dx == 0 {
            return Err(SwapError::ZeroAmount);
        }
        let x_in = self.reserves[i] as u128;
        let y_out = self.reserves[j] as u128;
        if x_in == 0 || y_out == 0 {
            return Err(SwapError::InsufficientLiquidity);
        }

        // No input fee; add dx to x_in, compute out, subtract 1, apply output fee.
        let x_new = x_in.checked_add(dx as u128).ok_or(SwapError::MathOverflow)?;
        if x_new == 0 {
            return Err(SwapError::MathOverflow);
        }
        let k = x_in.checked_mul(y_out).ok_or(SwapError::MathOverflow)?;
        let y_new = k / x_new;
        let mut out = y_out.checked_sub(y_new).ok_or(SwapError::MathOverflow)?;
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
