use std::collections::HashMap;

use nuts_derive::Storable;
use nuts_storable::{HasDims, Storable};
use nuts_storable::{ItemType, Value};

#[derive(Storable, Clone)]
struct InnerStats {
    value: f64,
    #[storable(dims("dim"))]
    draws: Vec<f64>,
}

#[derive(Storable, Clone)]
struct InnerStats2 {
    value2: f64,
    #[storable(dims("dim"))]
    draws2: Vec<f64>,
}

#[derive(Storable, Clone)]
struct ExampleStats {
    step_size: f64,
    n_steps: u64,
    is_adapting: bool,
    #[storable(dims("dim", "dim2"))]
    gradients: Vec<f64>,
    #[storable(dims("dim", "dim2"))]
    gradients2: Option<Vec<f64>>,
    #[storable(flatten)]
    inner: InnerStats,
    #[storable(flatten)]
    inner2: Option<InnerStats2>,
    #[storable(ignore)]
    _not_stored: String,
}

#[derive(Storable)]
struct Example2<P: HasDims, S: Storable<P>> {
    field1: u64,
    field2: S,
    #[storable(ignore)]
    _phantom: std::marker::PhantomData<fn() -> P>,
}

#[test]
fn test_storable() {
    struct Parent {}

    impl nuts_storable::HasDims for Parent {
        fn dim_sizes(&self) -> HashMap<String, u64> {
            HashMap::from([("dim".to_string(), 3), ("dim2".to_string(), 3)])
        }
    }

    let inner = InnerStats {
        value: 1.0,
        draws: vec![1.0, 2.0, 3.0],
    };
    let inner2 = InnerStats2 {
        value2: 8.0,
        draws2: vec![9.0, 2.0, 3.0],
    };
    let mut stats = ExampleStats {
        step_size: 0.1,
        n_steps: 10,
        is_adapting: true,
        gradients: vec![0.1, 0.2, 0.3],
        gradients2: None,
        inner,
        inner2: Some(inner2),
        _not_stored: "should not be stored".to_string(),
    };

    let mut stats2: Example2<Parent, _> = Example2 {
        field1: 42,
        field2: stats.clone(),
        _phantom: std::marker::PhantomData,
    };

    let parent = Parent {};

    assert_eq!(
        ExampleStats::names(&parent),
        vec![
            "step_size".to_string(),
            "n_steps".to_string(),
            "is_adapting".to_string(),
            "gradients".to_string(),
            "gradients2".to_string(),
            "value".to_string(),
            "draws".to_string(),
            "value2".to_string(),
            "draws2".to_string(),
        ]
    );

    assert_eq!(ExampleStats::item_type(&parent, "step_size"), ItemType::F64);
    assert_eq!(ExampleStats::item_type(&parent, "n_steps"), ItemType::U64);
    assert_eq!(
        ExampleStats::item_type(&parent, "is_adapting"),
        ItemType::Bool
    );
    assert_eq!(ExampleStats::item_type(&parent, "gradients"), ItemType::F64);

    assert_eq!(ExampleStats::dims(&parent, "step_size").len(), 0);
    assert_eq!(ExampleStats::dims(&parent, "n_steps").len(), 0);
    assert_eq!(ExampleStats::dims(&parent, "is_adapting").len(), 0);
    assert_eq!(
        ExampleStats::dims(&parent, "gradients"),
        vec!["dim".to_string(), "dim2".to_string()]
    );
    assert_eq!(
        ExampleStats::dims(&parent, "draws"),
        vec!["dim".to_string()]
    );

    let vals = stats.get_all(&parent);
    assert_eq!(vals.len(), 9);
    assert_eq!(vals[0].1, Some(Value::ScalarF64(0.1)));
    assert_eq!(vals[1].1, Some(Value::ScalarU64(10)));
    assert_eq!(vals[2].1, Some(Value::ScalarBool(true)));
    assert_eq!(vals[4].1, None);
    assert_eq!(vals[7].1, Some(Value::ScalarF64(8.0)));

    assert_eq!(
        Example2::<_, ExampleStats>::item_type(&parent, "step_size"),
        ItemType::F64
    );

    let vals2 = stats2.field2.get_all(&parent);
    assert_eq!(vals2.len(), 9);
}
