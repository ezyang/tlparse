use tlparse;
use std::path::Path;
use std::collections::HashMap;
use std::path::PathBuf;

fn prefix_exists(map: &HashMap<PathBuf, String>, prefix: &str) -> bool {
    map.keys().any(|key| key.to_str().map_or(false, |s| s.starts_with(prefix)))
}


#[test]
fn test_parse_simple() {
    let expected_files = [
        "0_0_0/aot_forward_graph.txt",
        "0_0_0/dynamo_output_graph.txt",
        "index.html",
        "0_0_0/inductor_post_grad_graph.txt",
        "0_0_0/inductor_output_code", // This always has an output hash, so we use prefixes instead of full
        "0_0_0/dynamo_guards.html"
    ];
    // Read the test file
    // simple.log was generated from the following:
    // TORCH_TRACE=~/trace_logs/test python test/inductor/test_torchinductor.py  -k test_custom_op_fixed_layout_channels_last_cpu
    let path = Path::new("tests/inputs/simple.log").to_path_buf();
    let config = tlparse::ParseConfig {
        strict: true,
    };
    let output = tlparse::parse_path(&path, config);
    assert!(output.is_ok());
    let map: HashMap<PathBuf, String> = output.unwrap().into_iter().collect();
    // Check all files are present
    for prefix in expected_files {
        assert!(prefix_exists(&map, prefix), "{} not found in output", prefix);
    }
}
