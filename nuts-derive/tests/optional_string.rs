use std::collections::HashMap;

use nuts_storable::{HasDims, ItemType, Storable, Value};

#[derive(Debug, Default)]
struct TestDims;

impl HasDims for TestDims {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::new()
    }
}

#[derive(Debug, nuts_derive::Storable)]
struct OptionalStringStats {
    diverging: bool,
    message: Option<String>,
}

#[derive(Debug, nuts_derive::Storable)]
struct StringAndVectorStats {
    message: Option<String>,
    #[storable(dims("unconstrained_parameter"))]
    values: Option<Vec<f64>>,
}

#[test]
fn derives_storable_for_optional_string_fields() {
    let parent = TestDims;

    assert_eq!(
        OptionalStringStats::names(&parent),
        vec!["diverging", "message"]
    );
    assert_eq!(
        OptionalStringStats::item_type(&parent, "diverging"),
        ItemType::Bool
    );
    assert_eq!(
        OptionalStringStats::item_type(&parent, "message"),
        ItemType::String
    );
    assert_eq!(
        OptionalStringStats::dims(&parent, "diverging"),
        Vec::<&str>::new()
    );
    assert_eq!(
        OptionalStringStats::dims(&parent, "message"),
        Vec::<&str>::new()
    );

    let mut stats = OptionalStringStats {
        diverging: true,
        message: Some("recoverable logp error".to_string()),
    };

    let values = stats.get_all(&parent);
    assert_eq!(values.len(), 2);
    assert_eq!(values[0].0, "diverging");
    assert_eq!(values[0].1, Some(Value::ScalarBool(true)));
    assert_eq!(values[1].0, "message");
    assert_eq!(
        values[1].1,
        Some(Value::ScalarString("recoverable logp error".to_string()))
    );
}

#[test]
fn optional_string_none_serializes_as_none() {
    let parent = TestDims;
    let mut stats = OptionalStringStats {
        diverging: false,
        message: None,
    };

    let values = stats.get_all(&parent);
    assert_eq!(values.len(), 2);
    assert_eq!(values[0].0, "diverging");
    assert_eq!(values[0].1, Some(Value::ScalarBool(false)));
    assert_eq!(values[1].0, "message");
    assert_eq!(values[1].1, None);
}

#[test]
fn optional_string_coexists_with_vector_fields() {
    let parent = TestDims;

    assert_eq!(
        StringAndVectorStats::names(&parent),
        vec!["message", "values"]
    );
    assert_eq!(
        StringAndVectorStats::item_type(&parent, "message"),
        ItemType::String
    );
    assert_eq!(
        StringAndVectorStats::item_type(&parent, "values"),
        ItemType::F64
    );
    assert_eq!(
        StringAndVectorStats::dims(&parent, "message"),
        Vec::<&str>::new()
    );
    assert_eq!(
        StringAndVectorStats::dims(&parent, "values"),
        vec!["unconstrained_parameter"]
    );

    let mut stats = StringAndVectorStats {
        message: Some("diverged".to_string()),
        values: Some(vec![1.0, 2.0, 3.0]),
    };

    let values = stats.get_all(&parent);
    assert_eq!(values.len(), 2);
    assert_eq!(values[0].0, "message");
    assert_eq!(
        values[0].1,
        Some(Value::ScalarString("diverged".to_string()))
    );
    assert_eq!(values[1].0, "values");
    assert_eq!(values[1].1, Some(Value::F64(vec![1.0, 2.0, 3.0])));
}
