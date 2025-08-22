use std::env;
use stableswap_pool::{StableSwapPool, SCALE_6, USDC, USDT};

fn main() {
    match env::args().nth(1).as_deref() {
        Some("sweep") => run_sweep(),
        _ => run_demo(),
    }
}

fn run_demo() {
    // 10,000 per side, A=100, fee=0.06%
    let mut pool = StableSwapPool::new([10_000 * SCALE_6, 10_000 * SCALE_6], 100);
    pool.fee_bps = 6;

    println!("USDC/USDT StableSwap demo");
    println!("A = {}, fee_bps = {}", pool.amplification_coefficient, pool.fee_bps);
    println!("reserves = [USDC: {}, USDT: {}]", pool.reserves[USDC], pool.reserves[USDT]);

    let d = pool.get_d();
    println!("D = {}", d);
    println!("spot: {:.8} USDT/USDC", pool.spot_price_usdc_in_usdt());

    // Non-mutating quotes only (no state change, no D-after)
    let trade_sizes = [1, 10, 100, 1000]; // USDC units
    for units in trade_sizes {
        let dx = units * SCALE_6;
        match pool.get_dy_usdc_to_usdt(dx) {
            Ok(dy) => {
                let bps_int = pool.calculate_slippage_bps(dx);
                let bps_f64 = pool.calculate_slippage_bps_f64(dx);
                println!(
                    "USDC→USDT: dx={} ({} USDC), dy={} (post-fee) | improvement: {} bps (~{:.3} bps)",
                    dx, units, dy, bps_int, bps_f64
                );
            }
            Err(e) => eprintln!("quote failed for {} USDC: {:?}", units, e),
        }
    }
}
