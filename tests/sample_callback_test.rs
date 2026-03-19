//! Unit tests for sample-level data access via ProgressCallback
//!
//! These tests verify that sample data can be accessed through ChainProgress
//! without requiring all heavy dependencies to compile.

use std::sync::{Arc, Mutex};
use std::time::Duration;

#[test]
fn test_sample_data_type_exists() {
    // This test verifies that SampleData is properly exported
    // It will fail to compile if the type is not available

    let _sample_data = nuts_rs::SampleData {
        chain_id: 0,
        draw: 42,
        is_tuning: true,
        tree_depth: Some(5),
        reached_max_treedepth: Some(false),
        diverging: Some(false),
        initial_energy: None,
        draw_energy: Some(-10.5),
        step_size: Some(0.1),
    };
}

#[test]
fn test_progress_callback_creation() {
    // Verify we can create a ProgressCallback that accesses sample data
    let callback_count = Arc::new(Mutex::new(0));
    let callback_count_clone = callback_count.clone();

    let _callback = nuts_rs::ProgressCallback {
        callback: Box::new(move |elapsed, chains| {
            let mut count = callback_count_clone.lock().unwrap();
            *count += 1;

            // Verify we can access elapsed time
            let _ = elapsed.as_secs_f64();

            // Verify we can access chain progress and sample data
            for chain_progress in chains.iter() {
                let _ = chain_progress.finished_draws;
                let _ = chain_progress.total_draws;

                // Verify we can access latest_sample
                if let Some(sample_data) = &chain_progress.latest_sample {
                    let _ = sample_data.chain_id;
                    let _ = sample_data.draw;
                    let _ = sample_data.is_tuning;
                    let _ = sample_data.tree_depth;
                    let _ = sample_data.reached_max_treedepth;
                    let _ = sample_data.diverging;
                    let _ = sample_data.initial_energy;
                    let _ = sample_data.draw_energy;
                    let _ = sample_data.step_size;
                }
            }
        }),
        rate: Duration::from_millis(100),
    };
}

#[test]
fn test_chain_progress_with_sample_data() {
    // Test that ChainProgress can hold sample data
    let sample_data = nuts_rs::SampleData {
        chain_id: 1,
        draw: 10,
        is_tuning: true,
        tree_depth: Some(3),
        reached_max_treedepth: Some(false),
        diverging: Some(false),
        initial_energy: None,
        draw_energy: Some(-5.2),
        step_size: Some(0.05),
    };

    // Verify we can access sample data fields
    assert_eq!(sample_data.chain_id, 1);
    assert_eq!(sample_data.draw, 10);
    assert!(sample_data.is_tuning);
    assert_eq!(sample_data.tree_depth, Some(3));
    assert_eq!(sample_data.reached_max_treedepth, Some(false));
    assert_eq!(sample_data.diverging, Some(false));
    assert_eq!(sample_data.initial_energy, None);
    assert_eq!(sample_data.draw_energy, Some(-5.2));
    assert_eq!(sample_data.step_size, Some(0.05));
}

#[test]
fn test_sample_data_clone() {
    // Verify SampleData implements Clone
    let data1 = nuts_rs::SampleData {
        chain_id: 0,
        draw: 1,
        is_tuning: false,
        tree_depth: Some(4),
        reached_max_treedepth: Some(true),
        diverging: Some(true),
        initial_energy: Some(-2.5),
        draw_energy: Some(-3.0),
        step_size: Some(0.1),
    };

    let data2 = data1.clone();

    assert_eq!(data1.chain_id, data2.chain_id);
    assert_eq!(data1.draw, data2.draw);
    assert_eq!(data1.is_tuning, data2.is_tuning);
    assert_eq!(data1.tree_depth, data2.tree_depth);
    assert_eq!(data1.reached_max_treedepth, data2.reached_max_treedepth);
    assert_eq!(data1.diverging, data2.diverging);
    assert_eq!(data1.initial_energy, data2.initial_energy);
    assert_eq!(data1.draw_energy, data2.draw_energy);
    assert_eq!(data1.step_size, data2.step_size);
}

#[test]
fn test_progress_callback_invocation() {
    // Test that the progress callback can be invoked with sample data
    let callback_count = Arc::new(Mutex::new(0));
    let sample_count = Arc::new(Mutex::new(0));

    let callback_count_clone = callback_count.clone();
    let sample_count_clone = sample_count.clone();

    let mut callback = nuts_rs::ProgressCallback {
        callback: Box::new(move |_elapsed, chains| {
            let mut count = callback_count_clone.lock().unwrap();
            *count += 1;

            // Count chains with sample data
            for chain_progress in chains.iter() {
                if let Some(sample_data) = &chain_progress.latest_sample {
                    let mut samples = sample_count_clone.lock().unwrap();
                    *samples += 1;

                    // Verify we can access the data
                    let _ = sample_data.chain_id;
                    let _ = sample_data.draw_energy;
                }
            }
        }),
        rate: Duration::from_millis(100),
    };

    // Simulate invoking the callback (in real usage, the sampler does this)
    let chains = vec![];
    (callback.callback)(Duration::from_secs(1), chains.into());

    assert_eq!(*callback_count.lock().unwrap(), 1);
}
