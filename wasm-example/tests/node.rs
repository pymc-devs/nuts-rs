//! Runs on wasm32-unknown-unknown in node with `wasm-pack test --node wasm-example`, and
//! natively with `cargo test -p wasm-example`.
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn posterior_mean_is_close() {
    let chains = wasm_example::sample(42).expect("sampling should succeed");
    assert_eq!(chains.len(), 2);
    for chain in &chains {
        assert_eq!(chain.posterior.num_rows(), 500);
    }

    // 2 chains x 500 draws x 3 dims of a unit normal: the standard error is about 0.03.
    let mean = wasm_example::mean_of_draws(&chains).unwrap();
    assert!(
        (mean - wasm_example::MEAN).abs() < 0.2,
        "posterior mean {mean}, expected about {}",
        wasm_example::MEAN
    );
}
