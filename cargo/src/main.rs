use stableswap_pool::{StableSwapPool, SCALE_6};

fn main() {
    let pool = StableSwapPool::new([10_000 * SCALE_6, 10_000 * SCALE_6], 300);
    println!("D = {}", pool.get_d());
    let dx = SCALE_6; // 1 unit
    match pool.get_dy(0, 1, dx) {
        Ok(dy) => println!("Swap 0→1: dx={}, dy={} (post-fee={}% bps)", dx, dy, pool.fee_bps),
        Err(e) => eprintln!("quote failed: {:?}", e),
    }
    println!("Improvement vs xy=k: {} bps", pool.calculate_slippage_bps(dx));
}
