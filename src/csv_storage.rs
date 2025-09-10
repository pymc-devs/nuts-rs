//! CSV storage backend for nuts-rs that outputs CmdStan-compatible CSV files
//!
//! This module provides a CSV storage backend that saves MCMC samples and
//! statistics in a format compatible with CmdStan, allowing existing Stan
//! analysis tools and libraries to read nuts-rs results.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use nuts_storable::{ItemType, Value};

use crate::storage::{ChainStorage, StorageConfig, TraceStorage};
use crate::{Math, Progress, Settings};

/// Configuration for CSV-based MCMC storage.
///
/// This storage backend creates Stan-compatible CSV files with one file per chain.
/// Files are named `chain_{id}.csv` where `{id}` is the chain number starting from 0.
///
/// The CSV format matches CmdStan output:
/// - Header row with column names
/// - Warmup samples with negative sample_id
/// - Post-warmup samples with positive sample_id
/// - Standard Stan statistics (lp__, stepsize, treedepth, etc.)
/// - Parameter columns
pub struct CsvConfig {
    /// Directory where CSV files will be written
    output_dir: PathBuf,
    /// Number of decimal places for floating point values
    precision: usize,
    /// Whether to store warmup samples (default: true)
    store_warmup: bool,
}

impl CsvConfig {
    /// Create a new CSV configuration.
    ///
    /// # Arguments
    ///
    /// * `output_dir` - Directory where CSV files will be written
    ///
    /// # Example
    ///
    /// ```rust
    /// use nuts_rs::CsvConfig;
    /// let config = CsvConfig::new("mcmc_output");
    /// ```
    pub fn new<P: AsRef<Path>>(output_dir: P) -> Self {
        Self {
            output_dir: output_dir.as_ref().to_path_buf(),
            precision: 6,
            store_warmup: true,
        }
    }

    /// Set the precision (number of decimal places) for floating point values.
    ///
    /// Default is 6 decimal places.
    pub fn with_precision(mut self, precision: usize) -> Self {
        self.precision = precision;
        self
    }

    /// Configure whether to store warmup samples.
    ///
    /// When true (default), warmup samples are included with negative sample IDs.
    /// When false, only post-warmup samples are stored.
    pub fn store_warmup(mut self, store: bool) -> Self {
        self.store_warmup = store;
        self
    }
}

/// Main CSV storage managing multiple chains
pub struct CsvTraceStorage {
    output_dir: PathBuf,
    precision: usize,
    store_warmup: bool,
    parameter_names: Vec<String>,
    column_mapping: Vec<(String, usize)>, // (data_name, index_in_data)
}

/// Per-chain CSV storage
pub struct CsvChainStorage {
    writer: BufWriter<File>,
    precision: usize,
    store_warmup: bool,
    parameter_names: Vec<String>,
    column_mapping: Vec<(String, usize)>, // (data_name, index_in_data)
    is_first_sample: bool,
    headers_written: bool,
}

impl CsvChainStorage {
    /// Create a new CSV chain storage
    fn new(
        output_dir: &Path,
        chain_id: u64,
        precision: usize,
        store_warmup: bool,
        parameter_names: Vec<String>,
        column_mapping: Vec<(String, usize)>,
    ) -> Result<Self> {
        std::fs::create_dir_all(output_dir)
            .with_context(|| format!("Failed to create output directory: {:?}", output_dir))?;

        let file_path = output_dir.join(format!("chain_{}.csv", chain_id));
        let file = File::create(&file_path)
            .with_context(|| format!("Failed to create CSV file: {:?}", file_path))?;
        let writer = BufWriter::new(file);

        Ok(Self {
            writer,
            precision,
            store_warmup,
            parameter_names,
            column_mapping,
            is_first_sample: true,
            headers_written: false,
        })
    }

    /// Write the CSV header row
    fn write_header(&mut self) -> Result<()> {
        if self.headers_written {
            return Ok(());
        }

        // Standard CmdStan header format - only the core columns
        let mut headers = vec![
            "lp__".to_string(),
            "accept_stat__".to_string(),
            "stepsize__".to_string(),
            "treedepth__".to_string(),
            "n_leapfrog__".to_string(),
            "divergent__".to_string(),
            "energy__".to_string(),
        ];

        // Add parameter columns from the expanded parameter vector
        for param_name in &self.parameter_names {
            headers.push(param_name.clone());
        }

        // Write header row
        writeln!(self.writer, "{}", headers.join(","))?;
        self.headers_written = true;
        Ok(())
    }

    /// Format a value for CSV output
    fn format_value(&self, value: &Value) -> String {
        match value {
            Value::ScalarF64(v) => {
                if v.is_nan() {
                    "NA".to_string()
                } else if v.is_infinite() {
                    if *v > 0.0 { "Inf" } else { "-Inf" }.to_string()
                } else {
                    format!("{:.prec$}", v, prec = self.precision)
                }
            }
            Value::ScalarF32(v) => {
                if v.is_nan() {
                    "NA".to_string()
                } else if v.is_infinite() {
                    if *v > 0.0 { "Inf" } else { "-Inf" }.to_string()
                } else {
                    format!("{:.prec$}", v, prec = self.precision)
                }
            }
            Value::ScalarU64(v) => v.to_string(),
            Value::ScalarI64(v) => v.to_string(),
            Value::ScalarBool(v) => if *v { "1" } else { "0" }.to_string(),
            Value::F64(vec) => {
                // For vector values, we'll just use the first element for now
                // A more sophisticated implementation would handle multi-dimensional parameters
                if vec.is_empty() {
                    "NA".to_string()
                } else {
                    self.format_value(&Value::ScalarF64(vec[0]))
                }
            }
            Value::F32(vec) => {
                if vec.is_empty() {
                    "NA".to_string()
                } else {
                    self.format_value(&Value::ScalarF32(vec[0]))
                }
            }
            Value::U64(vec) => {
                if vec.is_empty() {
                    "NA".to_string()
                } else {
                    vec[0].to_string()
                }
            }
            Value::I64(vec) => {
                if vec.is_empty() {
                    "NA".to_string()
                } else {
                    vec[0].to_string()
                }
            }
            Value::Bool(vec) => {
                if vec.is_empty() {
                    "NA".to_string()
                } else {
                    if vec[0] { "1" } else { "0" }.to_string()
                }
            }
            Value::ScalarString(v) => v.clone(),
            Value::Strings(vec) => {
                if vec.is_empty() {
                    "NA".to_string()
                } else {
                    vec[0].clone()
                }
            }
        }
    }

    /// Write a single sample row to the CSV file
    fn write_sample_row(
        &mut self,
        stats: &Vec<(&str, Option<Value>)>,
        draws: &Vec<(&str, Option<Value>)>,
        _info: &Progress,
    ) -> Result<()> {
        let mut row_values = Vec::new();

        // Create lookup maps for quick access
        let stats_map: HashMap<&str, &Option<Value>> = stats.iter().map(|(k, v)| (*k, v)).collect();
        let draws_map: HashMap<&str, &Option<Value>> = draws.iter().map(|(k, v)| (*k, v)).collect();

        // Helper function to get stat value
        let get_stat_value = |name: &str| -> String {
            stats_map
                .get(name)
                .and_then(|opt| opt.as_ref())
                .map(|v| self.format_value(v))
                .unwrap_or_else(|| "NA".to_string())
        };

        row_values.push(get_stat_value("logp"));
        row_values.push(get_stat_value("mean_tree_accept"));
        row_values.push(get_stat_value("step_size"));
        row_values.push(get_stat_value("depth"));
        row_values.push(get_stat_value("n_steps"));
        let divergent_val = stats_map
            .get("diverging")
            .and_then(|opt| opt.as_ref())
            .map(|v| match v {
                Value::ScalarBool(true) => "1".to_string(),
                Value::ScalarBool(false) => "0".to_string(),
                _ => "0".to_string(),
            })
            .unwrap_or_else(|| "0".to_string());
        row_values.push(divergent_val);

        row_values.push(get_stat_value("energy"));

        // Add parameter values using the column mapping
        for (_param_name, (data_name, index)) in
            self.parameter_names.iter().zip(&self.column_mapping)
        {
            if let Some(Some(data_value)) = draws_map.get(data_name.as_str()) {
                let formatted_value = match data_value {
                    Value::F64(vec) => {
                        if *index < vec.len() {
                            self.format_value(&Value::ScalarF64(vec[*index]))
                        } else {
                            "NA".to_string()
                        }
                    }
                    Value::F32(vec) => {
                        if *index < vec.len() {
                            self.format_value(&Value::ScalarF32(vec[*index]))
                        } else {
                            "NA".to_string()
                        }
                    }
                    Value::I64(vec) => {
                        if *index < vec.len() {
                            self.format_value(&Value::ScalarI64(vec[*index]))
                        } else {
                            "NA".to_string()
                        }
                    }
                    Value::U64(vec) => {
                        if *index < vec.len() {
                            self.format_value(&Value::ScalarU64(vec[*index]))
                        } else {
                            "NA".to_string()
                        }
                    }
                    // Handle scalar values (index should be 0)
                    scalar_val if *index == 0 => self.format_value(scalar_val),
                    _ => "NA".to_string(),
                };
                row_values.push(formatted_value);
            } else {
                row_values.push("NA".to_string());
            }
        }

        // Write the row
        writeln!(self.writer, "{}", row_values.join(","))?;
        Ok(())
    }
}

impl ChainStorage for CsvChainStorage {
    type Finalized = ();

    fn record_sample(
        &mut self,
        _settings: &impl Settings,
        stats: Vec<(&str, Option<Value>)>,
        draws: Vec<(&str, Option<Value>)>,
        info: &Progress,
    ) -> Result<()> {
        // Skip warmup samples if not storing them
        if info.tuning && !self.store_warmup {
            return Ok(());
        }

        // Write header on first sample
        if self.is_first_sample {
            self.write_header()?;
            self.is_first_sample = false;
        }

        self.write_sample_row(&stats, &draws, info)?;
        Ok(())
    }

    fn finalize(mut self) -> Result<Self::Finalized> {
        self.writer.flush().context("Failed to flush CSV file")?;
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        // BufWriter doesn't provide a way to flush without mutable reference
        // In practice, the buffer will be flushed when the file is closed
        Ok(())
    }
}

impl StorageConfig for CsvConfig {
    type Storage = CsvTraceStorage;

    fn new_trace<M: Math>(self, settings: &impl Settings, math: &M) -> Result<Self::Storage> {
        // Generate parameter names and column mapping using coordinates
        let (parameter_names, column_mapping) =
            generate_parameter_names_and_mapping(settings, math)?;

        Ok(CsvTraceStorage {
            output_dir: self.output_dir,
            precision: self.precision,
            store_warmup: self.store_warmup,
            parameter_names,
            column_mapping,
        })
    }
}

/// Generate parameter column names and mapping using coordinates or Stan-compliant indexing
fn generate_parameter_names_and_mapping<M: Math>(
    settings: &impl Settings,
    math: &M,
) -> Result<(Vec<String>, Vec<(String, usize)>)> {
    let data_dims = settings.data_dims_all(math);
    let coords = math.coords();
    let mut parameter_names = Vec::new();
    let mut column_mapping = Vec::new();

    for (var_name, var_dims) in data_dims {
        let data_type = settings.data_type(math, &var_name);

        // Only process vector types that could contain parameter values
        if matches!(
            data_type,
            ItemType::F64 | ItemType::F32 | ItemType::I64 | ItemType::U64
        ) {
            let (column_names, indices) = generate_column_names_and_indices_for_variable(
                &var_name, &var_dims, &coords, math,
            )?;

            for (name, index) in column_names.into_iter().zip(indices) {
                parameter_names.push(name);
                column_mapping.push((var_name.clone(), index));
            }
        }
    }

    // If no parameter names were generated, fall back to simple numbering
    if parameter_names.is_empty() {
        let dim_sizes = math.dim_sizes();
        let param_count = dim_sizes.get("expanded_parameter").unwrap_or(&0);
        for i in 0..*param_count {
            parameter_names.push(format!("param_{}", i + 1));
            // Try to find a data field that contains the parameters
            let data_names = settings.data_names(math);
            let mut found_field = false;
            for data_name in &data_names {
                let data_type = settings.data_type(math, data_name);
                if matches!(
                    data_type,
                    ItemType::F64 | ItemType::F32 | ItemType::I64 | ItemType::U64
                ) {
                    column_mapping.push((data_name.clone(), i as usize));
                    found_field = true;
                    break;
                }
            }
            if !found_field {
                column_mapping.push(("unknown".to_string(), i as usize));
            }
        }
    }

    Ok((parameter_names, column_mapping))
}

/// Generate column names and indices for a single variable using its dimensions and coordinates
fn generate_column_names_and_indices_for_variable<M: Math>(
    var_name: &str,
    var_dims: &[String],
    coords: &HashMap<String, Value>,
    math: &M,
) -> Result<(Vec<String>, Vec<usize>)> {
    let dim_sizes = math.dim_sizes();

    if var_dims.is_empty() {
        // Scalar variable
        return Ok((vec![var_name.to_string()], vec![0]));
    }

    // Check if we have meaningful coordinate names for all dimensions
    let has_meaningful_coords = var_dims.iter().all(|dim_name| {
        coords.get(dim_name).map_or(
            false,
            |coord_value| matches!(coord_value, Value::Strings(labels) if !labels.is_empty()),
        )
    });

    // Get coordinate labels for each dimension
    let mut dim_coords: Vec<Vec<String>> = Vec::new();
    let mut dim_sizes_vec: Vec<usize> = Vec::new();

    for dim_name in var_dims {
        let size = *dim_sizes.get(dim_name).unwrap_or(&1) as usize;
        dim_sizes_vec.push(size);

        if has_meaningful_coords {
            // Use coordinate names if available and meaningful
            if let Some(coord_value) = coords.get(dim_name) {
                match coord_value {
                    Value::Strings(labels) => {
                        dim_coords.push(labels.clone());
                    }
                    _ => {
                        // Fallback to 1-based indexing (Stan format)
                        dim_coords.push((1..=size).map(|i| i.to_string()).collect());
                    }
                }
            } else {
                // Fallback to 1-based indexing (Stan format)
                dim_coords.push((1..=size).map(|i| i.to_string()).collect());
            }
        } else {
            // Use Stan-compliant 1-based indexing
            dim_coords.push((1..=size).map(|i| i.to_string()).collect());
        }
    }

    // Generate Cartesian product using column-major order (Stan format)
    let (coord_names, indices) =
        cartesian_product_with_indices_column_major(&dim_coords, &dim_sizes_vec);

    // Prepend variable name to each coordinate combination
    let column_names: Vec<String> = coord_names
        .into_iter()
        .map(|coord| format!("{}.{}", var_name, coord))
        .collect();

    Ok((column_names, indices))
}

/// Compute the Cartesian product with column-major ordering (Stan format)
///
/// Stan uses what they call "column-major" ordering, but it's actually the same as
/// row-major order: the first index changes slowest, last index changes fastest.
/// For example, a 2x3 array produces: [1,1], [1,2], [1,3], [2,1], [2,2], [2,3]
fn cartesian_product_with_indices_column_major(
    coord_sets: &[Vec<String>],
    dim_sizes: &[usize],
) -> (Vec<String>, Vec<usize>) {
    if coord_sets.is_empty() {
        return (vec![], vec![]);
    }

    if coord_sets.len() == 1 {
        let indices: Vec<usize> = (0..coord_sets[0].len()).collect();
        return (coord_sets[0].clone(), indices);
    }

    let mut names = vec![];
    let mut indices = vec![];

    // Stan's "column-major" is actually row-major order
    cartesian_product_recursive_with_indices(
        coord_sets,
        dim_sizes,
        0,
        &mut String::new(),
        &mut vec![],
        &mut names,
        &mut indices,
    );

    (names, indices)
}

fn cartesian_product_recursive_with_indices(
    coord_sets: &[Vec<String>],
    dim_sizes: &[usize],
    dim_idx: usize,
    current_name: &mut String,
    current_indices: &mut Vec<usize>,
    result_names: &mut Vec<String>,
    result_indices: &mut Vec<usize>,
) {
    if dim_idx == coord_sets.len() {
        result_names.push(current_name.clone());
        // Compute linear index from multi-dimensional indices
        let mut linear_index = 0;
        for (i, &idx) in current_indices.iter().enumerate() {
            let mut stride = 1;
            for &size in &dim_sizes[i + 1..] {
                stride *= size;
            }
            linear_index += idx * stride;
        }
        result_indices.push(linear_index);
        return;
    }

    let is_first_dim = dim_idx == 0;

    for (coord_idx, coord) in coord_sets[dim_idx].iter().enumerate() {
        let mut new_name = current_name.clone();
        if !is_first_dim {
            new_name.push('.');
        }
        new_name.push_str(coord);

        current_indices.push(coord_idx);
        cartesian_product_recursive_with_indices(
            coord_sets,
            dim_sizes,
            dim_idx + 1,
            &mut new_name,
            current_indices,
            result_names,
            result_indices,
        );
        current_indices.pop();
    }
}

impl TraceStorage for CsvTraceStorage {
    type ChainStorage = CsvChainStorage;
    type Finalized = ();

    fn initialize_trace_for_chain(&self, chain_id: u64) -> Result<Self::ChainStorage> {
        CsvChainStorage::new(
            &self.output_dir,
            chain_id,
            self.precision,
            self.store_warmup,
            self.parameter_names.clone(),
            self.column_mapping.clone(),
        )
    }

    fn finalize(
        self,
        traces: Vec<Result<<Self::ChainStorage as ChainStorage>::Finalized>>,
    ) -> Result<(Option<anyhow::Error>, Self::Finalized)> {
        // Check for any errors in the chain finalizations
        for trace_result in traces {
            if let Err(err) = trace_result {
                return Ok((Some(err), ()));
            }
        }
        Ok((None, ()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CpuLogpFunc, CpuMath, CpuMathError, DiagGradNutsSettings, LogpError, Model, Sampler,
    };
    use anyhow::Result;
    use nuts_derive::Storable;
    use nuts_storable::{HasDims, Value};
    use rand::Rng;
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;
    use thiserror::Error;

    #[allow(dead_code)]
    #[derive(Debug, Error)]
    enum TestLogpError {
        #[error("Test error")]
        Test,
    }

    impl LogpError for TestLogpError {
        fn is_recoverable(&self) -> bool {
            false
        }
    }

    /// Test model with multi-dimensional coordinates
    #[derive(Clone)]
    struct MultiDimTestLogp {
        dim_a: usize,
        dim_b: usize,
    }

    impl HasDims for MultiDimTestLogp {
        fn dim_sizes(&self) -> HashMap<String, u64> {
            HashMap::from([
                ("a".to_string(), self.dim_a as u64),
                ("b".to_string(), self.dim_b as u64),
            ])
        }

        fn coords(&self) -> HashMap<String, Value> {
            HashMap::from([
                (
                    "a".to_string(),
                    Value::Strings(vec!["x".to_string(), "y".to_string()]),
                ),
                (
                    "b".to_string(),
                    Value::Strings(vec!["alpha".to_string(), "beta".to_string()]),
                ),
            ])
        }
    }

    #[derive(Storable)]
    struct MultiDimExpandedDraw {
        #[storable(dims("a", "b"))]
        param_matrix: Vec<f64>,
        scalar_value: f64,
    }

    impl CpuLogpFunc for MultiDimTestLogp {
        type LogpError = TestLogpError;
        type FlowParameters = ();
        type ExpandedVector = MultiDimExpandedDraw;

        fn dim(&self) -> usize {
            self.dim_a * self.dim_b
        }

        fn logp(&mut self, x: &[f64], grad: &mut [f64]) -> Result<f64, Self::LogpError> {
            let mut logp = 0.0;
            for (i, &xi) in x.iter().enumerate() {
                logp -= 0.5 * xi * xi;
                grad[i] = -xi;
            }
            Ok(logp)
        }

        fn expand_vector<R: Rng + ?Sized>(
            &mut self,
            _rng: &mut R,
            array: &[f64],
        ) -> Result<Self::ExpandedVector, CpuMathError> {
            Ok(MultiDimExpandedDraw {
                param_matrix: array.to_vec(),
                scalar_value: array.iter().sum(),
            })
        }

        fn vector_coord(&self) -> Option<Value> {
            Some(Value::Strings(
                (0..self.dim()).map(|i| format!("theta{}", i + 1)).collect(),
            ))
        }
    }

    struct MultiDimTestModel {
        math: CpuMath<MultiDimTestLogp>,
    }

    impl Model for MultiDimTestModel {
        type Math<'model>
            = CpuMath<MultiDimTestLogp>
        where
            Self: 'model;

        fn math(&self) -> Result<Self::Math<'_>> {
            Ok(self.math.clone())
        }

        fn init_position<R: Rng + ?Sized>(&self, rng: &mut R, position: &mut [f64]) -> Result<()> {
            for p in position.iter_mut() {
                *p = rng.random_range(-1.0..1.0);
            }
            Ok(())
        }
    }

    /// Test model without coordinates (fallback behavior)
    #[derive(Clone)]
    struct SimpleTestLogp {
        dim: usize,
    }

    impl HasDims for SimpleTestLogp {
        fn dim_sizes(&self) -> HashMap<String, u64> {
            HashMap::from([("simple_param".to_string(), self.dim as u64)])
        }
        // No coords() method - should use fallback
    }

    #[derive(Storable)]
    struct SimpleExpandedDraw {
        #[storable(dims("simple_param"))]
        values: Vec<f64>,
    }

    impl CpuLogpFunc for SimpleTestLogp {
        type LogpError = TestLogpError;
        type FlowParameters = ();
        type ExpandedVector = SimpleExpandedDraw;

        fn dim(&self) -> usize {
            self.dim
        }

        fn logp(&mut self, x: &[f64], grad: &mut [f64]) -> Result<f64, Self::LogpError> {
            let mut logp = 0.0;
            for (i, &xi) in x.iter().enumerate() {
                logp -= 0.5 * xi * xi;
                grad[i] = -xi;
            }
            Ok(logp)
        }

        fn expand_vector<R: Rng + ?Sized>(
            &mut self,
            _rng: &mut R,
            array: &[f64],
        ) -> Result<Self::ExpandedVector, CpuMathError> {
            Ok(SimpleExpandedDraw {
                values: array.to_vec(),
            })
        }

        fn vector_coord(&self) -> Option<Value> {
            Some(Value::Strings(vec![
                "param1".to_string(),
                "param2".to_string(),
                "param3".to_string(),
            ]))
        }
    }

    struct SimpleTestModel {
        math: CpuMath<SimpleTestLogp>,
    }

    impl Model for SimpleTestModel {
        type Math<'model>
            = CpuMath<SimpleTestLogp>
        where
            Self: 'model;

        fn math(&self) -> Result<Self::Math<'_>> {
            Ok(self.math.clone())
        }

        fn init_position<R: Rng + ?Sized>(&self, rng: &mut R, position: &mut [f64]) -> Result<()> {
            for p in position.iter_mut() {
                *p = rng.random_range(-1.0..1.0);
            }
            Ok(())
        }
    }

    fn read_csv_header(path: &Path) -> Result<String> {
        let content = fs::read_to_string(path)?;
        content
            .lines()
            .next()
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow::anyhow!("Empty CSV file"))
    }

    #[test]
    fn test_multidim_coordinate_naming() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let output_path = temp_dir.path().join("multidim_test");

        // Create model with 2x2 parameter matrix
        let model = MultiDimTestModel {
            math: CpuMath::new(MultiDimTestLogp { dim_a: 2, dim_b: 2 }),
        };

        let mut settings = DiagGradNutsSettings::default();
        settings.num_chains = 1;
        settings.num_tune = 10;
        settings.num_draws = 20;
        settings.seed = 42;

        let csv_config = CsvConfig::new(&output_path)
            .with_precision(6)
            .store_warmup(false);

        let mut sampler = Some(Sampler::new(model, settings, csv_config, 1, None)?);

        // Wait for sampling to complete
        while let Some(sampler_) = sampler.take() {
            match sampler_.wait_timeout(std::time::Duration::from_millis(100)) {
                crate::SamplerWaitResult::Trace(_) => break,
                crate::SamplerWaitResult::Timeout(s) => sampler = Some(s),
                crate::SamplerWaitResult::Err(err, _) => return Err(err),
            }
        }

        // Check that CSV file was created
        let csv_file = output_path.join("chain_0.csv");
        assert!(csv_file.exists());

        // Check header contains expected coordinate names
        let header = read_csv_header(&csv_file)?;

        // Should contain Cartesian product: x.alpha, x.beta, y.alpha, y.beta
        assert!(header.contains("param_matrix.x.alpha"));
        assert!(header.contains("param_matrix.x.beta"));
        assert!(header.contains("param_matrix.y.alpha"));
        assert!(header.contains("param_matrix.y.beta"));
        assert!(header.contains("scalar_value"));

        // Verify column order (Cartesian product should be in correct order)
        let columns: Vec<&str> = header.split(',').collect();
        let param_columns: Vec<&str> = columns
            .iter()
            .filter(|col| col.starts_with("param_matrix."))
            .cloned()
            .collect();

        assert_eq!(
            param_columns,
            vec![
                "param_matrix.x.alpha",
                "param_matrix.x.beta",
                "param_matrix.y.alpha",
                "param_matrix.y.beta"
            ]
        );

        Ok(())
    }

    #[test]
    fn test_fallback_coordinate_naming() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let output_path = temp_dir.path().join("simple_test");

        // Create model with 3 parameters but no coordinate specification
        let model = SimpleTestModel {
            math: CpuMath::new(SimpleTestLogp { dim: 3 }),
        };

        let mut settings = DiagGradNutsSettings::default();
        settings.num_chains = 1;
        settings.num_tune = 5;
        settings.num_draws = 10;
        settings.seed = 123;

        let csv_config = CsvConfig::new(&output_path)
            .with_precision(6)
            .store_warmup(false);

        let mut sampler = Some(Sampler::new(model, settings, csv_config, 1, None)?);

        // Wait for sampling to complete
        while let Some(sampler_) = sampler.take() {
            match sampler_.wait_timeout(std::time::Duration::from_millis(100)) {
                crate::SamplerWaitResult::Trace(_) => break,
                crate::SamplerWaitResult::Timeout(s) => sampler = Some(s),
                crate::SamplerWaitResult::Err(err, _) => return Err(err),
            }
        }

        // Check that CSV file was created
        let csv_file = output_path.join("chain_0.csv");
        assert!(csv_file.exists());

        // Check header uses fallback numeric naming
        let header = read_csv_header(&csv_file)?;

        // Should fall back to 1-based indices since no coordinates provided
        assert!(header.contains("values.1"));
        assert!(header.contains("values.2"));
        assert!(header.contains("values.3"));

        Ok(())
    }

    #[test]
    fn test_cartesian_product_generation() {
        let coord_sets = vec![
            vec!["x".to_string(), "y".to_string()],
            vec!["alpha".to_string(), "beta".to_string()],
        ];
        let dim_sizes = vec![2, 2];

        let (names, indices) = cartesian_product_with_indices_column_major(&coord_sets, &dim_sizes);

        assert_eq!(names, vec!["x.alpha", "x.beta", "y.alpha", "y.beta"]);

        assert_eq!(indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_single_dimension_coordinates() {
        let coord_sets = vec![vec!["param1".to_string(), "param2".to_string()]];
        let dim_sizes = vec![2];

        let (names, indices) = cartesian_product_with_indices_column_major(&coord_sets, &dim_sizes);

        assert_eq!(names, vec!["param1", "param2"]);
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn test_three_dimension_cartesian_product() {
        let coord_sets = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["1".to_string()],
            vec!["i".to_string(), "j".to_string()],
        ];
        let dim_sizes = vec![2, 1, 2];

        let (names, indices) = cartesian_product_with_indices_column_major(&coord_sets, &dim_sizes);

        assert_eq!(names, vec!["a.1.i", "a.1.j", "b.1.i", "b.1.j"]);

        assert_eq!(indices, vec![0, 1, 2, 3]);
    }
}
