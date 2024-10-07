use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use tlparse;

fn prefix_exists(map: &HashMap<PathBuf, String>, prefix: &str) -> bool {
    map.keys()
        .any(|key| key.to_str().map_or(false, |s| s.starts_with(prefix)))
}

#[test]
fn test_parse_simple() {
    let expected_files = [
        "0_0_0/aot_forward_graph",
        "0_0_0/dynamo_output_graph",
        "index.html",
        "failures_and_restarts.html",
        "0_0_0/inductor_post_grad_graph",
        "0_0_0/inductor_output_code",
        "0_0_0/dynamo_guards",
    ];
    // Read the test file
    // simple.log was generated from the following:
    // TORCH_TRACE=~/trace_logs/test python test/inductor/test_torchinductor.py  -k test_custom_op_fixed_layout_channels_last_cpu
    let path = Path::new("tests/inputs/simple.log").to_path_buf();
    let config = tlparse::ParseConfig {
        strict: true,
        ..Default::default()
    };
    let output = tlparse::parse_path(&path, config);
    assert!(output.is_ok());
    let map: HashMap<PathBuf, String> = output.unwrap().into_iter().collect();
    // Check all files are present
    for prefix in expected_files {
        assert!(
            prefix_exists(&map, prefix),
            "{} not found in output",
            prefix
        );
    }
}

#[test]
fn test_parse_compilation_metrics() {
    let expected_files = [
        "0_0_1/dynamo_output_graph",
        "0_0_1/dynamo_guards",
        "0_0_1/compilation_metrics",
        "1_0_1/dynamo_output_graph",
        "1_0_1/dynamo_guards",
        "1_0_1/compilation_metrics",
        "2_0_0/dynamo_output_graph",
        "2_0_0/dynamo_guards",
        "2_0_0/compilation_metrics",
        "index.html",
        "failures_and_restarts.html",
    ];
    // Read the test file
    // simple.log was generated from the following:
    // TORCH_TRACE=~/trace_logs/test python test/inductor/test_torchinductor.py  -k TORCH_TRACE=~/trace_logs/comp_metrics python test/dynamo/test_misc.py -k test_graph_break_compilation_metrics
    let path = Path::new("tests/inputs/comp_metrics.log").to_path_buf();
    let config = tlparse::ParseConfig {
        strict: true,
        ..Default::default()
    };
    let output = tlparse::parse_path(&path, config);
    assert!(output.is_ok());
    let map: HashMap<PathBuf, String> = output.unwrap().into_iter().collect();
    // Check all files are present
    for prefix in expected_files {
        assert!(
            prefix_exists(&map, prefix),
            "{} not found in output",
            prefix
        );
    }
}

#[test]
fn test_parse_compilation_failures() {
    let expected_files = [
        "0_0_0/dynamo_output_graph",
        "0_0_0/compilation_metrics",
        "index.html",
        "failures_and_restarts.html",
    ];
    // Read the test file
    // simple.log was generated from the following:
    // TORCH_TRACE=~/trace_logs/test python test/inductor/test_torchinductor.py  -k TORCH_TRACE=~/trace_logs/comp_metrics python test/dynamo/test_misc.py -k test_graph_break_compilation_metrics_on_failure
    let path = Path::new("tests/inputs/comp_failure.log").to_path_buf();
    let config = tlparse::ParseConfig {
        strict: true,
        ..Default::default()
    };
    let output = tlparse::parse_path(&path, config);
    assert!(output.is_ok());
    let map: HashMap<PathBuf, String> = output.unwrap().into_iter().collect();
    // Check all files are present
    for prefix in expected_files {
        assert!(
            prefix_exists(&map, prefix),
            "{} not found in output",
            prefix
        );
    }
}

#[test]
fn test_parse_artifact() {
    let expected_files = ["0_0_0/fx_graph_cache_hash", "index.html"];
    // Read the test file
    // simple.log was generated from the following:
    // TORCH_TRACE=~/trace_logs/test python test/inductor/test_torchinductor.py  -k TORCH_TRACE=~/trace_logs/comp_metrics python test/dynamo/test_misc.py -k test_graph_break_compilation_metrics_on_failure
    let path = Path::new("tests/inputs/artifacts.log").to_path_buf();
    let config = tlparse::ParseConfig {
        strict: true,
        ..Default::default()
    };
    let output = tlparse::parse_path(&path, config);
    assert!(output.is_ok());
    let map: HashMap<PathBuf, String> = output.unwrap().into_iter().collect();
    // Check all files are present
    for prefix in expected_files {
        assert!(
            prefix_exists(&map, prefix),
            "{} not found in output",
            prefix
        );
    }
}

#[test]
fn test_parse_chromium_event() {
    let expected_files = ["chromium_events.json", "index.html"];
    // Read the test file
    // simple.log was generated from the following:
    // TORCH_TRACE=~/trace_logs/test python test/inductor/test_torchinductor.py  -k TORCH_TRACE=~/trace_logs/comp_metrics python test/dynamo/test_misc.py -k test_graph_break_compilation_metrics_on_failure
    let path = Path::new("tests/inputs/chromium_events.log").to_path_buf();
    let config = tlparse::ParseConfig {
        strict: true,
        ..Default::default()
    };
    let output = tlparse::parse_path(&path, config);
    assert!(output.is_ok());
    let map: HashMap<PathBuf, String> = output.unwrap().into_iter().collect();
    // Check all files are present
    for prefix in expected_files {
        assert!(
            prefix_exists(&map, prefix),
            "{} not found in output",
            prefix
        );
    }
}

#[test]
fn test_cache_hit_miss() {
    let expected_files = [
        "1_0_0/fx_graph_cache_miss_8",
        "1_0_0/fx_graph_cache_hit_17",
        "index.html",
    ];
    // Generated via TORCH_TRACE=~/trace_logs/test python test/inductor/test_codecache.py -k test_flex_attention_caching
    let path = Path::new("tests/inputs/cache_hit_miss.log").to_path_buf();
    let config = tlparse::ParseConfig {
        strict: true,
        ..Default::default()
    };
    let output = tlparse::parse_path(&path, config);
    assert!(output.is_ok());
    let map: HashMap<PathBuf, String> = output.unwrap().into_iter().collect();
    // Check all files are present
    for prefix in expected_files {
        assert!(
            prefix_exists(&map, prefix),
            "{} not found in output",
            prefix
        );
    }
}
