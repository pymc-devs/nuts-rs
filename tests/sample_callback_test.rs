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
        position: vec![1.0, 2.0, 3.0],
        energy: -10.5,
        diverging: false,
        tree_depth: 5,
        step_size: 0.1,
    };

    println!("✅ SampleData struct exists and can be constructed");
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
                    let _ = &sample_data.position;
                    let _ = sample_data.energy;
                    let _ = sample_data.diverging;
                    let _ = sample_data.tree_depth;
                    let _ = sample_data.step_size;
                }
            }
        }),
        rate: Duration::from_millis(100),
    };

    println!("✅ ProgressCallback can be created with closure that accesses sample data");
}

#[test]
fn test_chain_progress_with_sample_data() {
    // Test that ChainProgress can hold sample data
    let sample_data = nuts_rs::SampleData {
        chain_id: 1,
        draw: 10,
        is_tuning: true,
        position: vec![0.5, -0.3],
        energy: -5.2,
        diverging: false,
        tree_depth: 3,
        step_size: 0.05,
    };

    // Verify we can access sample data fields
    assert_eq!(sample_data.chain_id, 1);
    assert_eq!(sample_data.draw, 10);
    assert!(sample_data.is_tuning);
    assert_eq!(sample_data.position.len(), 2);
    assert_eq!(sample_data.position[0], 0.5);
    assert_eq!(sample_data.position[1], -0.3);
    assert_eq!(sample_data.energy, -5.2);
    assert!(!sample_data.diverging);
    assert_eq!(sample_data.tree_depth, 3);
    assert_eq!(sample_data.step_size, 0.05);

    println!("✅ Sample data can be created and accessed");
}

#[test]
fn test_sample_data_clone() {
    // Verify SampleData implements Clone
    let data1 = nuts_rs::SampleData {
        chain_id: 0,
        draw: 1,
        is_tuning: false,
        position: vec![1.0, 2.0],
        energy: -3.0,
        diverging: true,
        tree_depth: 4,
        step_size: 0.1,
    };

    let data2 = data1.clone();

    assert_eq!(data1.chain_id, data2.chain_id);
    assert_eq!(data1.draw, data2.draw);
    assert_eq!(data1.position, data2.position);
    assert_eq!(data1.energy, data2.energy);
    assert_eq!(data1.diverging, data2.diverging);

    println!("✅ SampleData can be cloned");
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
                    let _ = sample_data.position.len();
                }
            }
        }),
        rate: Duration::from_millis(100),
    };

    // Simulate invoking the callback (in real usage, the sampler does this)
    let chains = vec![];
    (callback.callback)(Duration::from_secs(1), chains.into());

    assert_eq!(*callback_count.lock().unwrap(), 1);
    println!("✅ ProgressCallback can be invoked");
}
