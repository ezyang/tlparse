use anyhow::{anyhow, bail};
use chrono::Datelike;
use fxhash::{FxHashMap, FxHashSet};
use md5::{Digest, Md5};
use std::ffi::{OsStr, OsString};

use html_escape::encode_text;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use regex::Regex;
use serde_json::Value;
use std::cell::RefCell;
use std::fmt::Write as FmtWrite;
use std::fs::{self, File};
use std::io::{self, BufRead, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tinytemplate::TinyTemplate;

use crate::parsers::default_parsers;
use crate::parsers::ParserOutput;
use crate::parsers::StructuredLogParser;
use crate::templates::*;
use crate::types::*;
pub mod parsers;
mod templates;
mod types;
pub mod vllm;

pub use types::{
    ArtifactFlags, CollectiveSchedule, CollectivesParityReport, Diagnostics, DivergenceFlags,
    DivergenceGroup, ExecOrderSummary, GraphAnalysis, GraphCollectivesParity, GraphRuntime,
    MultiRankContext, OpRuntime, RankMetaData, RuntimeAnalysis, RuntimeRankDetail,
};

pub use execution_order::{
    analyze_execution_order, parse_graph_execution_order, ExecOrderIndexRow, ExecOrderIssue,
    ExecOrderReport,
};

#[derive(Debug)]
enum ParserResult {
    NoPayload,
    PayloadFilename(String),
}

pub struct ParseConfig {
    pub strict: bool,
    pub strict_compile_id: bool,
    pub custom_parsers: Vec<Box<dyn crate::parsers::StructuredLogParser>>,
    pub custom_header_html: String,
    pub verbose: bool,
    pub plain_text: bool,
    pub export: bool,
    pub inductor_provenance: bool,
}

impl Default for ParseConfig {
    fn default() -> Self {
        Self {
            strict: false,
            strict_compile_id: false,
            custom_parsers: Vec::default(),
            custom_header_html: String::default(),
            verbose: false,
            plain_text: false,
            export: false,
            inductor_provenance: false,
        }
    }
}

fn maybe_remove_convert_frame_suffixes(frames: &mut Vec<FrameSummary>) {
    let all_target_frames = [
        [
            ("torch/_dynamo/convert_frame.py", "catch_errors"),
            ("torch/_dynamo/convert_frame.py", "_convert_frame"),
            ("torch/_dynamo/convert_frame.py", "_convert_frame_assert"),
        ],
        [
            ("torch/_dynamo/convert_frame.py", "__call__"),
            ("torch/_dynamo/convert_frame.py", "__call__"),
            ("torch/_dynamo/convert_frame.py", "__call__"),
        ],
    ];

    let len = frames.len();
    for target_frames in all_target_frames {
        if len >= target_frames.len() {
            let suffix = &frames[len - target_frames.len()..];
            if suffix
                .iter()
                .zip(target_frames.iter())
                .all(|(frame, target)| {
                    simplify_filename(unintern_str(frame.filename).as_ref()) == target.0
                        && frame.name == target.1
                })
            {
                frames.truncate(len - target_frames.len());
            }
        }
    }
}

fn add_unique_suffix(raw_filename: PathBuf, output_count: i32) -> PathBuf {
    if let Some(stem) = raw_filename.file_stem() {
        let mut r = OsString::new();
        r.push(stem);
        r.push(OsStr::new("_"));
        r.push(output_count.to_string());
        if let Some(e) = raw_filename.extension() {
            r.push(OsStr::new("."));
            r.push(e);
        };
        raw_filename.with_file_name(r)
    } else {
        raw_filename
    }
}

fn add_file_output(
    filename: PathBuf,
    content: String,
    output: &mut ParseOutput,
    compile_directory: &mut Vec<OutputFile>,
    output_count: &mut i32,
    vllm_state: &vllm::VllmState,
    subgraph_name: &Option<String>,
) {
    let is_stack_traces = is_stack_traces_file(&filename);
    let maybe_content = if is_stack_traces {
        Some(content.clone())
    } else {
        None
    };
    output.push((filename.clone(), content));
    let filename_str = filename.to_string_lossy().to_string();

    let suffix = if filename_str.contains("cache_miss") {
        "❌".to_string()
    } else if filename_str.contains("cache_hit") {
        "✅".to_string()
    } else if filename_str.contains("cache_bypass") {
        "❓".to_string()
    } else {
        "".to_string()
    };

    // Track artifact for vLLM summary
    vllm_state.add_artifact(&filename, suffix.clone());

    let readable_url = if let Some(c) = maybe_content {
        Some(add_stack_traces_html(&filename, &c, output, output_count))
    } else {
        None
    };
    compile_directory.push(OutputFile {
        url: filename_str.clone(),
        name: filename_str,
        number: *output_count,
        suffix: suffix,
        readable_url,
        starts_group: false,
        group_name: subgraph_name.clone().unwrap_or_default(),
    });
    *output_count += 1;
}

fn is_stack_traces_file(path: &PathBuf) -> bool {
    if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
        name.starts_with("inductor_provenance_tracking_kernel_stack_traces")
            && name.ends_with(".json")
    } else {
        false
    }
}

fn add_stack_traces_html(
    json_path: &PathBuf,
    json_content: &str,
    output: &mut ParseOutput,
    output_count: &mut i32,
) -> String {
    let parsed: Value = match serde_json::from_str(json_content) {
        Ok(v) => v,
        Err(_) => return String::new(),
    };
    let mut html = String::from("<html><body>\n");
    if let Some(map) = parsed.as_object() {
        for (kernel, traces) in map {
            html.push_str(&format!("<h3>{}</h3>\n", encode_text(kernel)));
            if let Some(arr) = traces.as_array() {
                for t in arr {
                    if let Some(s) = t.as_str() {
                        // The JSON strings encode newlines as "\\n" sequences, so translate
                        // those into real line breaks for the HTML view.
                        let decoded = s.replace("\\n", "\n");
                        html.push_str("<pre>");
                        html.push_str(&encode_text(decoded.trim_end_matches('\n')));
                        html.push_str("</pre>\n");
                    }
                }
            }
        }
    }
    html.push_str("</body></html>\n");
    let mut html_path = json_path.clone();
    if let Some(stem) = json_path.file_stem().and_then(|s| s.to_str()) {
        html_path.set_file_name(format!("{stem}_readable.html"));
    } else {
        html_path.set_extension("html");
    }
    let html_path_str = html_path.to_string_lossy().to_string();
    output.push((html_path.clone(), html));
    *output_count += 1;
    html_path_str
}

fn run_parser<'t>(
    lineno: usize,
    parser: &Box<dyn StructuredLogParser + 't>,
    e: &Envelope,
    payload: &str,
    output_count: &mut i32,
    output: &mut ParseOutput,
    compile_directory: &mut Vec<OutputFile>,
    multi: &MultiProgress,
    stats: &mut Stats,
    vllm_state: &vllm::VllmState,
) -> ParserResult {
    let mut payload_filename = ParserResult::NoPayload;
    if let Some(md) = parser.get_metadata(&e) {
        let results = parser.parse(lineno, md, e.rank, &e.compile_id, &payload);
        match results {
            Ok(results) => {
                for parser_result in results {
                    match parser_result {
                        ParserOutput::File(raw_filename, out) => {
                            let filename = add_unique_suffix(raw_filename, *output_count);
                            add_file_output(
                                filename,
                                out,
                                output,
                                compile_directory,
                                output_count,
                                vllm_state,
                                &e.subgraph_name,
                            );
                        }
                        ParserOutput::GlobalFile(filename, out) => {
                            add_file_output(
                                filename,
                                out,
                                output,
                                compile_directory,
                                output_count,
                                vllm_state,
                                &e.subgraph_name,
                            );
                        }
                        ParserOutput::PayloadFile(raw_filename) => {
                            let filename = add_unique_suffix(raw_filename, *output_count);
                            payload_filename = ParserResult::PayloadFilename(
                                filename.to_string_lossy().to_string(),
                            );
                            add_file_output(
                                filename,
                                payload.to_string(),
                                output,
                                compile_directory,
                                output_count,
                                vllm_state,
                                &e.subgraph_name,
                            );
                        }
                        ParserOutput::PayloadReformatFile(raw_filename, formatter) => {
                            let filename = add_unique_suffix(raw_filename, *output_count);
                            match formatter(payload) {
                                Ok(formatted_content) => {
                                    payload_filename = ParserResult::PayloadFilename(
                                        filename.to_string_lossy().to_string(),
                                    );
                                    add_file_output(
                                        filename,
                                        formatted_content,
                                        output,
                                        compile_directory,
                                        output_count,
                                        vllm_state,
                                        &e.subgraph_name,
                                    );
                                }
                                Err(err) => {
                                    multi.suspend(|| {
                                        eprintln!(
                                            "Failed to format payload for {}: {}",
                                            filename.to_string_lossy(),
                                            err
                                        )
                                    });
                                    stats.fail_parser += 1;
                                }
                            }
                        }
                        ParserOutput::Link(name, url) => {
                            compile_directory.push(OutputFile {
                                url: url,
                                name: name,
                                number: *output_count,
                                suffix: "".to_string(),
                                readable_url: None,
                                starts_group: false,
                                group_name: e.subgraph_name.clone().unwrap_or_default(),
                            });
                            *output_count += 1;
                        }
                    }
                }
            }
            Err(err) => match parser.name() {
                "dynamo_guards" => {
                    multi.suspend(|| eprintln!("Failed to parse guards json: {}", err));
                    stats.fail_dynamo_guards_json += 1;
                }
                name => {
                    multi.suspend(|| eprintln!("Parser {name} failed: {err}"));
                    stats.fail_parser += 1;
                }
            },
        }
    }
    payload_filename
}

fn directory_to_json(
    directory: &FxIndexMap<Option<CompileId>, Vec<OutputFile>>,
) -> serde_json::Value {
    let mut json_map = serde_json::Map::new();

    for (compile_id, output_files) in directory {
        let key = compile_id
            .as_ref()
            .map_or_else(|| "unknown".to_string(), |cid| cid.to_string());

        let artifacts: Vec<serde_json::Value> = output_files
            .iter()
            .map(|file| {
                serde_json::json!({
                    "url": file.url,
                    // Strip away any leading directory names, that will just be in the url path anyway
                    "name": file.name.split('/').last().unwrap_or(&file.name),
                    "number": file.number,
                    "suffix": file.suffix,
                    "readable_url": file.readable_url,
                })
            })
            .collect();

        json_map.insert(key, serde_json::json!({"artifacts": artifacts}));
    }
    serde_json::Value::Object(json_map)
}

fn handle_guard(
    failure_type: &str,
    reason: &str,
    lineno: usize,
    e: &Envelope,
    payload: &str,
    output_count: &mut i32,
    output: &mut Vec<(PathBuf, String)>,
    compile_directory: &mut Vec<OutputFile>,
    multi: &MultiProgress,
    stats: &mut Stats,
    tt: &TinyTemplate,
    sym_expr_info_index: &RefCell<SymExprInfoIndex>,
    export_failures: &mut Vec<ExportFailure>,
    vllm_state: &vllm::VllmState,
) {
    let sym_expr_info_index_borrowed = sym_expr_info_index.borrow();
    let parser: Box<dyn StructuredLogParser> =
        Box::new(crate::parsers::PropagateRealTensorsParser {
            tt,
            sym_expr_info_index: &sym_expr_info_index_borrowed,
        });
    let _ = run_parser(
        lineno,
        &parser,
        e,
        payload,
        output_count,
        output,
        compile_directory,
        multi,
        stats,
        vllm_state,
    );

    let filename = format!(
        "symbolic_guard_information_{}.html",
        (*output_count - 1).to_string()
    );
    let compile_id_dir: PathBuf = e
        .compile_id
        .as_ref()
        .map_or(format!("unknown_{lineno}"), |cid| cid.as_directory_name())
        .into();
    let additional_info = format!(
        "Please click <a href='{}/{}'>here</a> for more information.",
        compile_id_dir.display(),
        filename,
    );

    export_failures.push(ExportFailure {
        failure_type: failure_type.to_string(),
        reason: reason.to_string(),
        additional_info,
    });
}

/// Write a JSON line to shortraw (raw.jsonl) by parsing into serde_json::Value,
/// inserting glog metadata fields (timestamp, thread, pathname, lineno), and
/// re-serializing with BTreeMap-sorted keys for deterministic output.
fn write_to_shortraw(
    content: &mut String,
    json_line: &str,
    payload_filename: Option<&str>,
    timestamp: &str,
    caps: &regex::Captures,
    multi: &MultiProgress,
    stats: &mut Stats,
) {
    let trimmed = json_line.trim_end();
    if !trimmed.ends_with('}') {
        multi.suspend(|| {
            eprintln!("JSON payload is not an object, dropping line from raw.jsonl");
        });
        stats.fail_json += 1;
        return;
    }

    // Parse as serde_json::Value (BTreeMap-backed) so keys are alphabetically sorted,
    // matching the baseline output format.
    let mut value: serde_json::Value = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(_) => {
            multi.suspend(|| {
                eprintln!("Failed to parse JSON for raw.jsonl, dropping line");
            });
            stats.fail_json += 1;
            return;
        }
    };

    let obj = value.as_object_mut().unwrap();

    // Check for key conflicts after parsing, so we check real keys not string patterns in values.
    let conflict_keys: &[&str] = if payload_filename.is_some() {
        &[
            "timestamp",
            "thread",
            "pathname",
            "lineno",
            "payload_filename",
        ]
    } else {
        &["timestamp", "thread", "pathname", "lineno"]
    };
    for key in conflict_keys {
        if obj.contains_key(*key) {
            multi.suspend(|| {
                eprintln!(
                    "Key conflict: \"{}\" already exists in JSON payload, skipping raw.jsonl JSONL conversion",
                    key
                );
            });
            stats.fail_key_conflict += 1;
            return;
        }
    }

    let thread = caps.name("thread").unwrap().as_str();
    let pathname = caps.name("pathname").unwrap().as_str();
    let lineno_str = caps.name("line").unwrap().as_str();

    // Parse lineno as a number to match baseline (it was inserted as raw numeric in old code)
    let lineno_value: serde_json::Value = if let Ok(n) = lineno_str.parse::<i64>() {
        serde_json::Value::Number(n.into())
    } else {
        serde_json::Value::String(lineno_str.to_string())
    };

    // Parse thread as a number to match baseline
    let thread_value: serde_json::Value = if let Ok(n) = thread.parse::<i64>() {
        serde_json::Value::Number(n.into())
    } else {
        serde_json::Value::String(thread.to_string())
    };

    obj.insert(
        "timestamp".to_string(),
        serde_json::Value::String(timestamp.to_string()),
    );
    obj.insert("thread".to_string(), thread_value);
    obj.insert(
        "pathname".to_string(),
        serde_json::Value::String(pathname.to_string()),
    );
    obj.insert("lineno".to_string(), lineno_value);

    if let Some(pf) = payload_filename {
        obj.insert(
            "payload_filename".to_string(),
            serde_json::Value::String(pf.to_string()),
        );
    }

    // Serialize with sorted keys (BTreeMap guarantees alphabetical order)
    content.push_str(&serde_json::to_string(&value).unwrap());
    content.push('\n');
}

pub fn parse_path(path: &PathBuf, config: &ParseConfig) -> anyhow::Result<ParseOutput> {
    let strict = config.strict;
    if !path.is_file() {
        bail!("{} is not a file", path.display())
    }
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    let file_size = metadata.len();

    // TODO: abstract out this spinner to not be part of the library
    // Instead, add a callback trait for CLIs to implement
    let multi = MultiProgress::new();
    let pb = multi.add(ProgressBar::new(file_size));
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} [{bytes_per_sec}] ({eta})")?
        .progress_chars("#>-"));
    let spinner = multi.add(ProgressBar::new_spinner());

    let is_gzipped = path.extension().map_or(false, |ext| ext == "gz");
    let reader: Box<dyn io::Read> = if is_gzipped {
        Box::new(flate2::read::GzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let reader = io::BufReader::new(reader);

    let re_glog = Regex::new(concat!(
        r"(?<level>[VIWEC])(?<month>\d{2})(?<day>\d{2}) ",
        r"(?<hour>\d{2}):(?<minute>\d{2}):(?<second>\d{2}).(?<millisecond>\d{6}) ",
        r"(?<thread>\d+)",
        r"(?<pathname>[^:]+):(?<line>\d+)\] ",
        r"(?<payload>.)"
    ))?;

    // Compute year once instead of calling chrono::Utc::now().year() per line
    let year = chrono::Utc::now().year();

    // Helper function to format timestamp as ISO-8601
    let format_timestamp = |caps: &regex::Captures| -> String {
        let month: u32 = caps.name("month").unwrap().as_str().parse().unwrap();
        let day: u32 = caps.name("day").unwrap().as_str().parse().unwrap();
        let hour: u32 = caps.name("hour").unwrap().as_str().parse().unwrap();
        let minute: u32 = caps.name("minute").unwrap().as_str().parse().unwrap();
        let second: u32 = caps.name("second").unwrap().as_str().parse().unwrap();
        let microsecond: u32 = caps.name("millisecond").unwrap().as_str().parse().unwrap();

        // Format as ISO-8601 with microsecond precision
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}Z",
            year, month, day, hour, minute, second, microsecond
        )
    };

    let mut stack_trie = StackTrieNode::default();
    let mut unknown_stack_trie = StackTrieNode::default();

    let mut stats = Stats::default();
    let _mod_count: FxHashMap<String, i32> = FxHashMap::default();

    let mut bytes_read: u64 = 0;

    // Some stuff for profiling
    let mut fastest_time = std::time::Duration::MAX;
    let mut slowest_time = std::time::Duration::ZERO;

    let mut expected_rank: Option<Option<u32>> = None;

    // Each entry is a compile id => (link, rendered name, output number)
    // For files, link and rendered name are the same
    // For links, you can specify a custom name for the link
    let mut directory: FxIndexMap<Option<CompileId>, Vec<OutputFile>> = FxIndexMap::default();

    let mut metrics_index: CompilationMetricsIndex = FxIndexMap::default();
    let stack_index: RefCell<StackIndex> = RefCell::new(FxHashMap::default());

    let symbolic_shape_specialization_index: RefCell<SymbolicShapeSpecializationIndex> =
        RefCell::new(FxHashMap::default());
    let guard_added_fast_index: RefCell<GuardAddedFastIndex> = RefCell::new(FxHashMap::default());
    let sym_expr_info_index: RefCell<SymExprInfoIndex> = RefCell::new(FxHashMap::default());
    let create_symbol_index: RefCell<CreateSymbolIndex> = RefCell::new(FxHashMap::default());
    let unbacked_symbol_index: RefCell<UnbackedSymbolIndex> = RefCell::new(FxHashMap::default());

    // Store results in an output ParseOutput
    let mut output: ParseOutput = Vec::new();

    // Store raw.jsonl content (without payloads)
    // Pre-allocate: shortraw is typically ~12% of raw log size
    let mut shortraw_content = String::with_capacity(file_size as usize / 8);

    let mut tt: TinyTemplate = TinyTemplate::new();
    tt.add_formatter("format_unescaped", tinytemplate::format_unescaped);
    tt.add_formatter("format_float", |value, output| {
        if let serde_json::Value::Number(n) = value {
            if let Some(f) = n.as_f64() {
                write!(output, "{:.3}", f)?;
                return Ok(());
            }
        }
        write!(output, "{}", value)?;
        Ok(())
    });
    if config.export {
        tt.add_template("index.html", TEMPLATE_EXPORT_INDEX)?;
        tt.add_template(
            "symbolic_guard_information.html",
            TEMPLATE_SYMBOLIC_GUARD_INFO,
        )?;
    } else {
        tt.add_template("index.html", TEMPLATE_INDEX)?;
        tt.add_template("failures_and_restarts.html", TEMPLATE_FAILURES_AND_RESTARTS)?;
        tt.add_template("dynamo_guards.html", TEMPLATE_DYNAMO_GUARDS)?;
        tt.add_template("compilation_metrics.html", TEMPLATE_COMPILATION_METRICS)?;
        tt.add_template(
            "bwd_compilation_metrics.html",
            TEMPLATE_BWD_COMPILATION_METRICS,
        )?;
        tt.add_template(
            "aot_autograd_backward_compilation_metrics.html",
            TEMPLATE_AOT_AUTOGRAD_BACKWARD_COMPILATION_METRICS,
        )?;
    }
    tt.add_template("provenance_tracking.html", TEMPLATE_PROVENANCE_TRACKING)?;
    tt.add_template("vllm_summary.html", vllm::templates::VLLM_SUMMARY_TEMPLATE)?;

    let mut unknown_fields: FxHashSet<String> = FxHashSet::default();

    let mut output_count = 0;

    let mut breaks = RestartsAndFailuresContext {
        css: TEMPLATE_FAILURES_CSS,
        failures: Vec::new(),
        qps: TEMPLATE_QUERY_PARAM_SCRIPT,
    };

    let mut export_failures: Vec<ExportFailure> = Vec::new();

    // NB: Sometimes, the log output we get from Logarithm stutters with a blank line.
    // Filter them out, they're never valid (a blank line in payload will still be \t)
    let mut iter = reader
        .lines()
        .enumerate()
        .filter_map(|(i, l)| match l {
            // 1-indexed line numbers please
            Ok(l) if !l.is_empty() => Some((i + 1, l)),
            _ => None,
        })
        .peekable();

    let default_parsers = default_parsers(&tt, config);
    let vllm_state = vllm::VllmState::new();
    let vllm_parsers = vllm::vllm_parsers_with_state(vllm_state.clone());
    let mut all_parsers: Vec<&Box<dyn StructuredLogParser>> = default_parsers.iter().collect();
    all_parsers.extend(vllm_parsers.iter());
    let mut chromium_events: Vec<serde_json::Value> = Vec::new();
    all_parsers.extend(config.custom_parsers.iter());

    // Reuse payload buffer across iterations to avoid repeated allocation
    let mut payload_buf = String::new();

    while let Some((lineno, line)) = iter.next() {
        bytes_read += line.len() as u64;
        pb.set_position(bytes_read);
        spinner.set_message(format!("{}", stats));
        //spinner.set_message(format!("{:?} {:?}", slowest_time, fastest_time));
        let start = Instant::now();

        let Some(caps) = re_glog.captures(&line) else {
            multi.suspend(|| eprintln!("Failed to parse glog prefix on line {}", lineno));
            stats.fail_glog += 1;
            continue;
        };

        let end = start.elapsed();
        if end < fastest_time {
            fastest_time = end;
        }
        if end > slowest_time {
            slowest_time = end;
        }
        let json_line = &line[caps.name("payload").unwrap().start()..];

        // Parse Envelope from JSON line (single parse -- no separate Value parse needed)
        let e = match serde_json::from_str::<Envelope>(json_line) {
            Ok(r) => r,
            Err(err) => {
                multi.suspend(|| {
                    eprintln!(
                        "Failed to parse metadata JSON: \n{:?} on line {}",
                        err, lineno
                    );
                });
                stats.fail_json += 1;
                // Best-effort shortraw write using the raw JSON string
                write_to_shortraw(
                    &mut shortraw_content,
                    json_line,
                    None,
                    &format_timestamp(&caps),
                    &caps,
                    &multi,
                    &mut stats,
                );
                continue;
            }
        };

        stats.unknown += e._other.len() as u64;

        for k in e._other.keys() {
            unknown_fields.insert(k.clone());
            if config.verbose {
                multi.suspend(|| eprintln!("Unknown field {}", k))
            }
        }

        if let Some((s, i)) = e.str {
            let mut intern_table = INTERN_TABLE.lock().unwrap();
            intern_table.insert(i, s);
            continue;
        };

        payload_buf.clear();
        if let Some(ref expect) = e.has_payload {
            let mut first = true;
            while let Some((_payload_lineno, payload_line)) =
                iter.next_if(|(_, l)| l.starts_with('\t'))
            {
                // Careful! Distinguish between missing EOL and not
                if !first {
                    payload_buf.push('\n');
                }
                first = false;
                payload_buf.push_str(&payload_line[1..]);
            }
            let mut hasher = Md5::new();
            hasher.update(&payload_buf);
            let hash = hasher.finalize();
            let mut expect_buf = [0u8; 16];
            if base16ct::lower::decode(expect, &mut expect_buf).is_ok() {
                if expect_buf != hash[..] {
                    // TODO: error log
                    stats.fail_payload_md5 += 1;
                }
            } else {
                stats.fail_payload_md5 += 1;
            }
        }

        match expected_rank {
            Some(rank) => {
                if rank != e.rank {
                    stats.other_rank += 1;
                    write_to_shortraw(
                        &mut shortraw_content,
                        json_line,
                        None,
                        &format_timestamp(&caps),
                        &caps,
                        &multi,
                        &mut stats,
                    );
                    continue;
                }
            }
            None => {
                // Allow logs with no rank and then some rank to be processed
                // Logs with no rank may be initialized before distributed rank is set
                if e.rank.is_some() {
                    multi.suspend(|| {
                        eprintln!("Detected rank: {:?}", e.rank);
                    });
                    expected_rank = Some(e.rank);
                }
            }
        };

        stats.ok += 1;

        // Some runtime compile ids don't have attempts. Collapse these entries into
        // attempt 0 for now.
        let mut compile_id_entry = e.compile_id.clone();
        if let Some(ref mut entry) = compile_id_entry {
            entry.normalize_attempt();
        }

        // TODO: output should be able to generate this without explicitly creating
        let compile_directory = directory.entry(compile_id_entry).or_default();

        let mut parser_payload_filename = ParserResult::NoPayload;
        for parser in &all_parsers {
            let result = run_parser(
                lineno,
                parser,
                &e,
                &payload_buf,
                &mut output_count,
                &mut output,
                compile_directory,
                &multi,
                &mut stats,
                &vllm_state,
            );
            // Take the last PayloadFilename entry as per the requirement
            if matches!(result, ParserResult::PayloadFilename(_)) {
                parser_payload_filename = result;
            }
        }

        if let Some(ref m) = e.compilation_metrics {
            let compile_id_dir: PathBuf = e
                .compile_id
                .as_ref()
                .map_or(format!("unknown_{lineno}"), |cid| cid.as_directory_name())
                .into();
            // Step 1: construct parser borrowing compile_directory immutably, call parse().
            // The parser + its borrow are dropped at the end of this block.
            let metrics_parse_result = {
                let parser = crate::parsers::CompilationMetricsParser {
                    tt: &tt,
                    stack_index: &stack_index,
                    symbolic_shape_specialization_index: &symbolic_shape_specialization_index,
                    guard_added_fast_index: &guard_added_fast_index,
                    create_symbol_index: &create_symbol_index,
                    unbacked_symbol_index: &unbacked_symbol_index,
                    output_files: compile_directory.as_slice(),
                    compile_id_dir: &compile_id_dir,
                };
                parser
                    .get_metadata(&e)
                    .map(|md| parser.parse(lineno, md, e.rank, &e.compile_id, &payload_buf))
            };
            // Step 2: parser is dropped, immutable borrow of compile_directory ends.
            // Now we can mutate compile_directory to add results.
            if let Some(result) = metrics_parse_result {
                match result {
                    Ok(results) => {
                        for parser_result in results {
                            match parser_result {
                                ParserOutput::File(raw_filename, out) => {
                                    let filename = add_unique_suffix(raw_filename, output_count);
                                    add_file_output(
                                        filename,
                                        out,
                                        &mut output,
                                        compile_directory,
                                        &mut output_count,
                                        &vllm_state,
                                        &e.subgraph_name,
                                    );
                                }
                                ParserOutput::GlobalFile(filename, out) => {
                                    add_file_output(
                                        filename,
                                        out,
                                        &mut output,
                                        compile_directory,
                                        &mut output_count,
                                        &vllm_state,
                                        &e.subgraph_name,
                                    );
                                }
                                ParserOutput::PayloadFile(raw_filename) => {
                                    let filename = add_unique_suffix(raw_filename, output_count);
                                    parser_payload_filename = ParserResult::PayloadFilename(
                                        filename.to_string_lossy().to_string(),
                                    );
                                    add_file_output(
                                        filename,
                                        payload_buf.to_string(),
                                        &mut output,
                                        compile_directory,
                                        &mut output_count,
                                        &vllm_state,
                                        &e.subgraph_name,
                                    );
                                }
                                ParserOutput::PayloadReformatFile(raw_filename, formatter) => {
                                    let filename = add_unique_suffix(raw_filename, output_count);
                                    match formatter(&payload_buf) {
                                        Ok(formatted_content) => {
                                            parser_payload_filename = ParserResult::PayloadFilename(
                                                filename.to_string_lossy().to_string(),
                                            );
                                            add_file_output(
                                                filename,
                                                formatted_content,
                                                &mut output,
                                                compile_directory,
                                                &mut output_count,
                                                &vllm_state,
                                                &e.subgraph_name,
                                            );
                                        }
                                        Err(err) => {
                                            multi.suspend(|| {
                                                eprintln!(
                                                    "Failed to format payload for {}: {}",
                                                    filename.to_string_lossy(),
                                                    err
                                                )
                                            });
                                            stats.fail_parser += 1;
                                        }
                                    }
                                }
                                ParserOutput::Link(name, url) => {
                                    compile_directory.push(OutputFile {
                                        url,
                                        name,
                                        number: output_count,
                                        suffix: "".to_string(),
                                        readable_url: None,
                                        starts_group: false,
                                        group_name: e.subgraph_name.clone().unwrap_or_default(),
                                    });
                                    output_count += 1;
                                }
                            }
                        }
                    }
                    Err(err) => {
                        multi.suspend(|| eprintln!("Parser compilation_metrics failed: {err}"));
                        stats.fail_parser += 1;
                    }
                }
            }

            // compilation metrics is always the last output, since it just ran
            let metrics_filename = format!(
                "compilation_metrics_{}.html",
                (output_count - 1).to_string(),
            );

            let id = e.compile_id.clone().map_or("(unknown) ".to_string(), |c| {
                format!(
                    "<a href='{}/{}'>{cid}</a> ",
                    compile_id_dir.display(),
                    metrics_filename,
                    cid = c,
                )
            });
            if let Some(rr) = m.restart_reasons.as_ref() {
                for restart in rr {
                    breaks.failures.push((
                        id.clone(),
                        format!("{}", FailureReason::Restart(restart.clone())),
                    ));
                }
            }
            if let Some(f) = m.fail_type.as_ref() {
                let reason = m
                    .fail_reason
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("Fail reason not found"))?;
                let user_frame_filename = m
                    .fail_user_frame_filename
                    .clone()
                    .unwrap_or(String::from("N/A"));
                let user_frame_lineno = m.fail_user_frame_lineno.unwrap_or(0);
                let failure_reason = FailureReason::Failure((
                    f.clone(),
                    reason.clone(),
                    user_frame_filename.clone(),
                    user_frame_lineno.clone(),
                ));
                breaks
                    .failures
                    .push((id.clone(), format!("{failure_reason}")));
            }
            let mut cid = e.compile_id.clone();
            if let Some(c) = cid.as_mut() {
                c.collapse_attempt();
            }
            metrics_index.entry(cid).or_default().push(m.clone());
        }

        if config.export {
            if let Some(ref guard) = e.guard_added {
                if guard.prefix.as_deref() != Some("eval") {
                    write_to_shortraw(
                        &mut shortraw_content,
                        json_line,
                        None,
                        &format_timestamp(&caps),
                        &caps,
                        &multi,
                        &mut stats,
                    );
                    continue;
                }
                let failure_type = "Guard Evaluated";

                let reason = format!(
                    "When exporting, the following guard was evaluated <code>{}</code>. This
                    might've resulted in a constraint violation error.",
                    guard.expr.clone().unwrap(),
                );

                handle_guard(
                    failure_type,
                    &reason,
                    lineno,
                    &e,
                    &payload_buf,
                    &mut output_count,
                    &mut output,
                    compile_directory,
                    &multi,
                    &mut stats,
                    &tt,
                    &sym_expr_info_index,
                    &mut export_failures,
                    &vllm_state,
                );
            }

            if let Some(ref guard) = e.propagate_real_tensors_provenance {
                let failure_type = "Data Dependent Error";

                let reason = format!(
                    "When exporting, we were unable to figure out if the
                    expression <code>{}</code> always holds.<br> As a result, it
                    was specialized to evaluate to <code>{}</code>, and asserts
                    were inserted into the graph.",
                    guard.expr.clone().unwrap(),
                    guard.result.clone().unwrap()
                );

                handle_guard(
                    failure_type,
                    &reason,
                    lineno,
                    &e,
                    &payload_buf,
                    &mut output_count,
                    &mut output,
                    compile_directory,
                    &multi,
                    &mut stats,
                    &tt,
                    &sym_expr_info_index,
                    &mut export_failures,
                    &vllm_state,
                );
            }

            if let Some(fake_kernel) = e.missing_fake_kernel {
                let failure_type = "Missing Fake Kernel";

                let reason = format!(
                    "<code>torch.ops.{}</code> is missing a fake kernel implementation",
                    fake_kernel.op.unwrap()
                );

                let additional_info = "Please refer to <a href='https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ahugy69p2jmz'>this doc</a> for more detailed instructions on how to write a fake kernel.";

                export_failures.push(ExportFailure {
                    failure_type: failure_type.to_string(),
                    reason: reason,
                    additional_info: additional_info.to_string(),
                });
            }

            if let Some(fake_kernel) = e.mismatched_fake_kernel {
                let failure_type = "Mismatched Fake Kernel";

                let reason = format!(
                    "<code>torch.ops.{}</code> has a fake kernel implementation,
                    but it has incorrect behavior, based on the real kernel.<br>
                    The reason for the mismatch is: {}",
                    fake_kernel.op.unwrap(),
                    fake_kernel.reason.unwrap(),
                );

                let additional_info = "Please refer to <a href='https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ahugy69p2jmz'>this doc</a> for more detailed instructions on how to write a fake kernel.";

                export_failures.push(ExportFailure {
                    failure_type: failure_type.to_string(),
                    reason: reason,
                    additional_info: additional_info.to_string(),
                });
            }

            if let Some(sym_expr_info) = e.expression_created {
                sym_expr_info_index
                    .borrow_mut()
                    .insert(sym_expr_info.result_id.unwrap(), sym_expr_info);
            }

            if let Some(ref unbacked_symbol) = e.create_unbacked_symbol {
                sym_expr_info_index.borrow_mut().insert(
                    unbacked_symbol.node_id.unwrap(),
                    SymExprInfoMetadata {
                        result: unbacked_symbol.symbol.clone(),
                        result_id: unbacked_symbol.node_id.clone(),
                        user_stack: unbacked_symbol.user_stack.clone(),
                        stack: unbacked_symbol.stack.clone(),
                        ..Default::default()
                    },
                );
            }
        }

        // Handle symbol creation events OUTSIDE of export block - they should always be collected
        if let Some(unbacked_symbol) = e.create_unbacked_symbol.clone() {
            let mut cid = e.compile_id.clone();
            if let Some(c) = cid.as_mut() {
                c.normalize_attempt();
            }
            unbacked_symbol_index
                .borrow_mut()
                .entry(cid)
                .or_default()
                .push(unbacked_symbol);
        }

        // Handle create_symbol events (backed symbols with concrete values)
        if let Some(symbol) = e.create_symbol.clone() {
            let mut cid = e.compile_id.clone();
            if let Some(c) = cid.as_mut() {
                c.normalize_attempt();
            }
            create_symbol_index
                .borrow_mut()
                .entry(cid)
                .or_default()
                .push(symbol);
        }

        if let Some(stack) = e.stack {
            unknown_stack_trie.insert(stack.clone(), None);
        }

        if let Some(_) = e.chromium_event {
            // Skip bad json in chromium event. This can happen if log lines are dropped.
            match serde_json::from_str(&payload_buf) {
                Ok(event) => chromium_events.push(event),
                Err(_) => {
                    // Continue processing instead of crashing
                    // If json line is dropped, we should see fail_payload_md5 in result because the
                    // payload doesn't match the md5.
                }
            }
        }

        if let Some(specialization) = e.symbolic_shape_specialization {
            symbolic_shape_specialization_index
                .borrow_mut()
                .entry(e.compile_id.clone())
                .or_default()
                .push(specialization);
        }
        if let Some(guard_added_fast) = e.guard_added_fast {
            guard_added_fast_index
                .borrow_mut()
                .entry(e.compile_id.clone())
                .or_default()
                .push(guard_added_fast)
        }

        if let Some(m) = e.dynamo_start {
            if let Some(mut stack) = m.stack {
                maybe_remove_convert_frame_suffixes(&mut stack);
                stack_index
                    .borrow_mut()
                    .insert(e.compile_id.clone(), stack.clone());
                stack_trie.insert(stack, e.compile_id.clone());
            };
        };

        // Handle payload file writing and determine final payload filename, but skip chromium events
        let final_payload_filename = match parser_payload_filename {
            ParserResult::PayloadFilename(filename) => Some(filename),
            ParserResult::NoPayload => {
                if let Some(ref expect) = e.has_payload {
                    // Only write payload file if no parser generated PayloadFile/PayloadReformatFile output and not a chromium event
                    if !payload_buf.is_empty() && e.chromium_event.is_none() {
                        let hash_str = expect;
                        let payload_path = PathBuf::from(format!("payloads/{}.txt", hash_str));
                        output.push((payload_path, payload_buf.clone()));
                        Some(format!("payloads/{}.txt", hash_str))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        };

        // Write to raw.jsonl with optional payload filename, but skip chromium events
        if e.chromium_event.is_none() {
            write_to_shortraw(
                &mut shortraw_content,
                json_line,
                final_payload_filename.as_deref(),
                &format_timestamp(&caps),
                &caps,
                &multi,
                &mut stats,
            );
        }
    }

    if config.export {
        let num_failures = export_failures.len();

        let exported_program_url = directory
            .values()
            .flatten()
            .find(|output_file| output_file.url.contains("exported_program"))
            .map(|output_file| output_file.url.clone());

        let index_context = ExportIndexContext {
            css: EXPORT_CSS,
            javascript: JAVASCRIPT,
            custom_header_html: config.custom_header_html.clone(),
            directory: directory
                .drain(..)
                .map(|(x, y)| (x.map_or("(unknown)".to_string(), |e| e.to_string()), y))
                .collect(),
            failures: export_failures,
            num_failures: num_failures,
            success: num_failures == 0,
            exported_program_url: exported_program_url.unwrap_or("".to_string()),
            qps: TEMPLATE_QUERY_PARAM_SCRIPT,
        };

        output.push((
            PathBuf::from("index.html"),
            tt.render("index.html", &index_context)?,
        ));

        return Ok(output);
    }

    output.push((
        PathBuf::from("failures_and_restarts.html"),
        tt.render("failures_and_restarts.html", &breaks)?,
    ));
    pb.finish_with_message("done");
    spinner.finish();

    output.push((
        PathBuf::from("chromium_events.json"),
        serde_json::to_string_pretty(&chromium_events).unwrap(),
    ));

    eprintln!("{}", stats);
    if unknown_fields.len() > 0 {
        eprintln!(
            "Unknown fields: {:?} (consider updating tlparse to render these)",
            unknown_fields
        );
    }

    let has_unknown_compile_id = directory.contains_key(&None);

    for files in directory.values_mut() {
        let mut prev_group = String::new();
        for file in files.iter_mut() {
            if file.group_name != prev_group {
                file.starts_group = true;
                prev_group = file.group_name.clone();
            }
        }
    }

    let directory_names: Vec<String> = directory
        .iter()
        .map(|(x, _)| {
            x.as_ref()
                .map_or("(unknown)".to_string(), |e| e.as_directory_name())
        })
        .collect();
    output.push((
        PathBuf::from("compile_directory.json"),
        serde_json::to_string_pretty(&directory_to_json(&directory))?,
    ));

    // Generate traditional tlparse index
    let index_context = IndexContext {
        css: CSS,
        javascript: JAVASCRIPT,
        custom_header_html: config.custom_header_html.clone(),
        directory: directory
            .drain(..)
            .map(|(x, y)| (x.map_or("(unknown)".to_string(), |e| e.to_string()), y))
            .collect(),
        stack_trie_html: stack_trie
            .fmt(Some(&metrics_index), "Stack", false)
            .unwrap(),
        unknown_stack_trie_html: unknown_stack_trie
            .fmt(Some(&metrics_index), "Stack", false)
            .unwrap(),
        has_unknown_stack_trie: !unknown_stack_trie.is_empty(),
        num_breaks: breaks.failures.len(),
        has_chromium_events: !chromium_events.is_empty(),
        qps: TEMPLATE_QUERY_PARAM_SCRIPT,
        has_inductor_provenance: config.inductor_provenance,
        directory_names: directory_names.clone(),
    };
    let tlparse_index_html = tt.render("index.html", &index_context)?;

    if vllm_state.has_artifacts() {
        // If vLLM artifacts are present, use vLLM summary as index.html and
        // save traditional tlparse index as tlparse_index.html for reference.
        // `has_vllm_artifacts` gets set to true when the vLLM parsers are
        // triggered. This happens when we see the following events:
        // `vllm_subgraph_*`, `vllm_compilation_config`,
        // `vllm_piecewise_split_graph`.
        let vllm_html = vllm::generate_vllm_summary(&vllm_state, &tt, &config.custom_header_html)?;
        output.push((PathBuf::from("index.html"), vllm_html));
        output.push((PathBuf::from("tlparse_index.html"), tlparse_index_html));
    } else {
        output.push((PathBuf::from("index.html"), tlparse_index_html));
    }

    // Include raw.log in output so library callers get it too.
    // For gzip inputs, decompress so the output is always plain text.
    if is_gzipped {
        let file = File::open(path)?;
        let mut decoder = flate2::read::GzDecoder::new(file);
        let mut raw_content = String::new();
        decoder.read_to_string(&mut raw_content)?;
        output.push((PathBuf::from("raw.log"), raw_content));
    } else {
        output.push((PathBuf::from("raw.log"), fs::read_to_string(path)?));
    }

    // Create string table from INTERN_TABLE as an array with nulls for missing indices
    let intern_table = INTERN_TABLE.lock().unwrap();
    let max_index = intern_table.keys().max().copied().unwrap_or(0) as usize;
    let mut string_table: Vec<Option<String>> = vec![None; max_index + 1];
    for (&index, value) in intern_table.iter() {
        string_table[index as usize] = Some(value.clone());
    }
    drop(intern_table); // Release the lock early

    // Serialize string table as JSON object
    let string_table_json = serde_json::json!({
        "string_table": string_table
    });
    let string_table_line = serde_json::to_string(&string_table_json)?;

    // Prepend string table to raw.jsonl content
    let mut final_shortraw_content =
        String::with_capacity(string_table_line.len() + 1 + shortraw_content.len());
    final_shortraw_content.push_str(&string_table_line);
    final_shortraw_content.push('\n');
    final_shortraw_content.push_str(&shortraw_content);

    output.push((PathBuf::from("raw.jsonl"), final_shortraw_content));

    // other_rank is included here because you should only have logs from one rank when
    // configured properly
    if strict
        && (stats.fail_glog
            + stats.fail_json
            + stats.fail_payload_md5
            + stats.other_rank
            + stats.fail_dynamo_guards_json
            + stats.fail_parser
            > 0)
    {
        // Report something went wrong
        return Err(anyhow!("Something went wrong"));
    }

    if config.strict_compile_id && has_unknown_compile_id {
        return Err(anyhow!("Some log entries did not have compile id"));
    }

    if config.inductor_provenance {
        // Helper function to get file content for a specific directory name
        fn get_file_content(
            output: &[(PathBuf, String)],
            filename_patterns: &[&str],
            directory_name: &str,
        ) -> String {
            // Try each pattern in order and return the first match found
            for pattern in filename_patterns {
                if let Some((_, content)) = output.iter().rev().find(|(path, _)| {
                    path.to_string_lossy()
                        .contains(&format!("{}/{}", directory_name, pattern))
                }) {
                    return content.clone();
                }
            }
            String::default()
        }

        // Generate HTML for each directory name
        for directory_name in &directory_names {
            let pre_grad_graph_content = get_file_content(
                &output,
                &["before_pre_grad_graph", "inductor_pre_grad_graph"],
                directory_name,
            );
            let post_grad_graph_content = get_file_content(
                &output,
                &["after_post_grad_graph", "inductor_post_grad_graph"],
                directory_name,
            );
            let output_code_content =
                get_file_content(&output, &["inductor_output_code"], directory_name);
            let aot_code_content =
                get_file_content(&output, &["inductor_aot_wrapper_code"], directory_name);
            let node_mappings_content = get_file_content(
                &output,
                &["inductor_provenance_tracking_node_mappings"],
                directory_name,
            );

            // Convert node mappings to line number mappings
            let line_mappings_content = convert_node_mappings_to_line_numbers(
                &node_mappings_content,
                &pre_grad_graph_content,
                &post_grad_graph_content,
                &output_code_content,
                &aot_code_content,
            );
            let line_mappings_content_str = serde_json::to_string_pretty(&line_mappings_content)
                .unwrap_or_else(|_| "{}".to_string());

            output.push((
                PathBuf::from(format!("provenance_tracking_{}.html", directory_name)),
                tt.render(
                    "provenance_tracking.html",
                    &ProvenanceContext {
                        css: PROVENANCE_CSS,
                        js: PROVENANCE_JS,
                        pre_grad_graph_content,
                        post_grad_graph_content,
                        output_code_content,
                        aot_code_content,
                        line_mappings_content: line_mappings_content_str,
                    },
                )?,
            ));
        }
    }

    Ok(output)
}

pub fn read_chromium_events_with_pid(
    path: &std::path::Path,
    rank_num: u32,
) -> anyhow::Result<Vec<serde_json::Value>> {
    use std::fs;

    if !path.exists() {
        return Ok(Vec::new());
    }

    let file_content = fs::read_to_string(path)?;

    match serde_json::from_str::<Vec<serde_json::Value>>(&file_content) {
        Ok(mut events) => {
            for event in &mut events {
                if let Some(obj) = event.as_object_mut() {
                    obj.insert("pid".to_string(), serde_json::json!(rank_num));
                }
            }
            Ok(events)
        }
        Err(_) => Ok(Vec::new()),
    }
}

pub fn generate_multi_rank_html(
    out_path: &PathBuf,
    sorted_ranks: Vec<String>,
    cfg: &ParseConfig,
    has_chromium_events: bool,
    show_desync_warning: bool,
    compile_id_divergence: bool,
    diagnostics: Diagnostics,
) -> anyhow::Result<(PathBuf, String)> {
    // Create the TinyTemplate instance for rendering the landing page.
    let mut tt = TinyTemplate::new();
    tt.add_formatter("format_unescaped", tinytemplate::format_unescaped);
    tt.add_template("multi_rank_index.html", TEMPLATE_MULTI_RANK_INDEX)?;

    let ctx = MultiRankContext {
        css: CSS,
        custom_header_html: &cfg.custom_header_html,
        num_ranks: sorted_ranks.len(),
        ranks: sorted_ranks,
        qps: TEMPLATE_QUERY_PARAM_SCRIPT,
        has_chromium_events,
        show_desync_warning,
        compile_id_divergence,
        diagnostics,
    };
    let html = tt.render("multi_rank_index.html", &ctx)?;
    let landing_page_path = out_path.join("index.html");

    Ok((landing_page_path, html))
}

/// Generate multi-rank landing and aggregated artifacts from pre-parsed rank outputs.
///
/// This mirrors the diagnostics and file emission performed by the CLI `--all-ranks-html` path.
/// It expects that per-rank outputs already exist under `out_dir/rank_<n>` and will:
/// - Combine Chromium events into `chromium_events.json` (if any)
/// - Write `runtime_estimations.json` and `chromium_trace_with_runtime.json` (if any)
/// - Write `collective_schedules.json` (if any) and per-rank `collectives_parity.json`
/// - Analyze runtime deltas and exec order summary
/// - Render `index.html` landing page using the same template and context
pub fn generate_multi_rank_landing(
    cfg: &ParseConfig,
    ctx: &MultiRankContext,
    out_dir: &Path,
) -> anyhow::Result<PathBuf> {
    use std::fs;

    // Parse ranks from context and ensure sorted numerically like the CLI
    let mut rank_nums: Vec<u32> = ctx
        .ranks
        .iter()
        .filter_map(|s| s.parse::<u32>().ok())
        .collect();
    rank_nums.sort_unstable();
    let sorted_ranks: Vec<String> = rank_nums.iter().map(|r| r.to_string()).collect();

    let out_path: PathBuf = out_dir.to_path_buf();

    // Collect compile ids and cache sequences per rank, and chromium events at root
    let mut rank_metadata: FxHashMap<u32, RankMetaData> = FxHashMap::default();
    let mut all_chromium_events: Vec<serde_json::Value> = Vec::new();

    for &rank_num in &rank_nums {
        let subdir = out_path.join(format!("rank_{rank_num}"));
        let chromium_events_path = subdir.join("chromium_events.json");
        let compile_dir_json = subdir.join("compile_directory.json");

        // extract compile IDs and cache sequence from compile_directory.json
        let mut compile_ids: FxHashSet<String> = FxHashSet::default();
        let content = fs::read_to_string(&compile_dir_json)?;
        let mut artifact_entries: Vec<(u64, String)> = Vec::new();

        if let Ok(serde_json::Value::Object(map)) =
            serde_json::from_str::<serde_json::Value>(&content)
        {
            for (key, val) in map.iter() {
                if key != "unknown" && !key.starts_with("unknown_") {
                    compile_ids.insert(key.clone());
                }
                if let Some(arr) = val.get("artifacts").and_then(|v| v.as_array()) {
                    for art in arr {
                        let suffix = art.get("suffix").and_then(|s| s.as_str()).unwrap_or("");
                        if suffix.is_empty() {
                            continue;
                        }
                        if let Some(num) = art.get("number").and_then(|n| n.as_u64()) {
                            artifact_entries.push((num, suffix.to_string()));
                        }
                    }
                }
            }
        }

        artifact_entries.sort_by_key(|(n, _)| *n);
        let cache_sequence: String = artifact_entries.into_iter().map(|(_, s)| s).collect();

        rank_metadata.insert(
            rank_num,
            RankMetaData {
                rank: rank_num,
                compile_ids,
                cache_sequence,
            },
        );

        // collect chromium events for each rank
        if chromium_events_path.exists() {
            let events = read_chromium_events_with_pid(&chromium_events_path, rank_num)?;
            all_chromium_events.extend(events);
        }
    }

    // Determine if there is any divergence in compile IDs across ranks
    let compile_id_divergence =
        if let Some(first) = rank_nums.iter().filter_map(|r| rank_metadata.get(r)).next() {
            rank_metadata
                .values()
                .any(|md| md.compile_ids != first.compile_ids)
        } else {
            false
        };

    // Group ranks by their cache hit/miss sequence
    let cache_seq_groups: FxHashMap<String, Vec<u32>> =
        rank_metadata
            .into_values()
            .fold(FxHashMap::default(), |mut acc, md| {
                acc.entry(md.cache_sequence).or_default().push(md.rank);
                acc
            });

    // Build groups describing cache hit/miss patterns per rank
    let cache_divergence_groups: Vec<DivergenceGroup> = if cache_seq_groups.len() > 1 {
        cache_seq_groups
            .iter()
            .map(|(seq, ranks_vec)| {
                let mut sorted = ranks_vec.clone();
                sorted.sort_unstable();
                DivergenceGroup {
                    sequence: seq.clone(),
                    ranks: sorted
                        .iter()
                        .map(|r| r.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    // combine chromium events from all ranks
    if !all_chromium_events.is_empty() {
        let combined_chromium_path = out_path.join("chromium_events.json");
        let combined_events_json = serde_json::to_string_pretty(&all_chromium_events)?;
        fs::write(combined_chromium_path, combined_events_json)?;
    }

    // Process runtime estimations from all ranks
    let runtime_estimations = crate::parsers::read_runtime_estimations(&out_path, &rank_nums)?;
    if !runtime_estimations.is_empty() {
        let runtime_path = out_path.join("runtime_estimations.json");
        fs::write(
            &runtime_path,
            serde_json::to_string_pretty(&runtime_estimations)?,
        )?;
        println!("Runtime estimations: {}", runtime_path.display());

        // Generate runtime trace events in a single pass
        let mut runtime_events: Vec<serde_json::Value> = Vec::new();
        let mut pid_set: FxHashSet<u32> = FxHashSet::default();
        let mut thread_names: FxHashMap<(u32, u32), String> = FxHashMap::default();

        // Concise, deterministic 32-bit TID from (rank, graph)
        let calc_tid = |rank: u32, graph: &str| -> u32 {
            use std::hash::{Hash, Hasher};
            let mut h = fxhash::FxHasher::default();
            (rank, graph).hash(&mut h);
            (h.finish() & 0xFFFF_FFFF) as u32
        };

        for gr in &runtime_estimations {
            pid_set.insert(gr.rank);
            let tid = calc_tid(gr.rank, &gr.graph);
            thread_names
                .entry((gr.rank, tid))
                .or_insert_with(|| gr.graph.clone());

            let mut time_offset_us: u64 = 0;
            for op in &gr.ops {
                let dur_us = (op.estimated_runtime_ns / 1000.0).ceil().max(1.0) as u64;
                runtime_events.push(serde_json::json!({
                    "name": op.name,
                    "ph": "X",
                    "ts": time_offset_us,
                    "dur": dur_us,
                    "pid": gr.rank,
                    "tid": tid,
                    "cat": "runtime",
                    "args": {
                        "graph": gr.graph,
                        "rank": gr.rank,
                        "runtime_ns": op.estimated_runtime_ns as u64
                    }
                }));
                time_offset_us += dur_us;
            }
        }

        let mut all_events: Vec<serde_json::Value> = runtime_events;

        // Emit process (rank) metadata in ascending pid order
        let mut pids: Vec<u32> = pid_set.into_iter().collect();
        pids.sort_unstable();
        for pid in pids.into_iter() {
            all_events.extend([
                serde_json::json!({
                    "name": "process_name",
                    "ph": "M",
                    "pid": pid,
                    "args": {"name": format!("Rank {}", pid)}
                }),
                serde_json::json!({
                    "name": "process_sort_index",
                    "ph": "M",
                    "pid": pid,
                    "args": {"sort_index": pid as i64}
                }),
            ]);
        }

        // Emit thread names sorted by graph name within each pid
        let mut threads_by_pid: FxHashMap<u32, Vec<(u32, String)>> = FxHashMap::default();
        for ((pid, tid), graph_name) in thread_names.into_iter() {
            threads_by_pid
                .entry(pid)
                .or_default()
                .push((tid, graph_name));
        }
        let mut pids_for_threads: Vec<u32> = threads_by_pid.keys().copied().collect();
        pids_for_threads.sort_unstable();
        for pid in pids_for_threads {
            let entries = threads_by_pid.remove(&pid).unwrap_or_default();
            for (idx, (tid, graph_name)) in entries.into_iter().enumerate() {
                all_events.extend([
                    serde_json::json!({
                        "name": "thread_name",
                        "ph": "M",
                        "pid": pid,
                        "tid": tid,
                        "args": {"name": format!("graph {}", graph_name)}
                    }),
                    serde_json::json!({
                        "name": "thread_sort_index",
                        "ph": "M",
                        "pid": pid,
                        "tid": tid,
                        "args": {"sort_index": idx as i64}
                    }),
                ]);
            }
        }

        fs::write(
            out_path.join("chromium_trace_with_runtime.json"),
            serde_json::to_string_pretty(&all_events)?,
        )?;
    }

    // Analyze graph runtime deltas across ranks
    let runtime_analysis = if !runtime_estimations.is_empty() {
        analyze_graph_runtime_deltas(&runtime_estimations)
    } else {
        None
    };

    // Process collective schedules from all ranks
    let collective_schedules = crate::parsers::read_collective_schedules(&out_path, &rank_nums)?;
    if !collective_schedules.is_empty() {
        let schedules_path = out_path.join("collective_schedules.json");
        fs::write(
            &schedules_path,
            serde_json::to_string_pretty(&collective_schedules)?,
        )?;
        println!("Collective schedules: {}", schedules_path.display());
    }

    crate::parsers::check_collectives_parity(&out_path, &rank_nums)?;

    // Process tensor meta fingerprints from all ranks
    let tensor_meta = crate::parsers::read_tensor_meta_fingerprints(&out_path, &rank_nums)?;
    let mut tensor_meta_groups: FxHashMap<String, Vec<u32>> = FxHashMap::default();
    if !tensor_meta.is_empty() {
        use std::collections::HashMap;
        // rank -> sorted list of (graph_id, fingerprint)
        let mut by_rank: HashMap<u32, Vec<(String, String)>> = HashMap::new();
        for tm in &tensor_meta {
            by_rank
                .entry(tm.rank)
                .or_default()
                .push((tm.graph.clone(), tm.fingerprint.clone()));
        }
        for (&rank, entries) in &mut by_rank {
            // sort by graph id to make cross-rank concatenation consistent
            let mut entries_sorted = entries.clone();
            entries_sorted.sort_by(|a, b| a.0.cmp(&b.0));
            let signature = entries_sorted
                .into_iter()
                .map(|(_, fp)| fp)
                .collect::<Vec<_>>()
                .join(",");
            tensor_meta_groups.entry(signature).or_default().push(rank);
        }
    }

    let tensor_meta_divergence_groups: Vec<DivergenceGroup> = if tensor_meta_groups.len() > 1 {
        tensor_meta_groups
            .iter()
            .map(|(seq, ranks_vec)| {
                let mut sorted = ranks_vec.clone();
                sorted.sort_unstable();
                DivergenceGroup {
                    sequence: seq.clone(),
                    ranks: sorted
                        .iter()
                        .map(|r| r.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    // Group ranks by their collective op sequence
    let mut collective_seq_groups: FxHashMap<String, Vec<u32>> = FxHashMap::default();
    if !collective_schedules.is_empty() {
        for &rank in &rank_nums {
            let ops_concat: String = collective_schedules
                .iter()
                .filter(|s| s.rank == rank)
                .flat_map(|s| s.ops.clone())
                .collect::<Vec<_>>()
                .join(",");
            collective_seq_groups
                .entry(ops_concat)
                .or_default()
                .push(rank);
        }
    }

    let collective_divergence_groups: Vec<DivergenceGroup> = if collective_seq_groups.len() > 1 {
        collective_seq_groups
            .iter()
            .map(|(seq, ranks_vec)| {
                let mut sorted = ranks_vec.clone();
                sorted.sort_unstable();
                DivergenceGroup {
                    sequence: seq.clone(),
                    ranks: sorted
                        .iter()
                        .map(|r| r.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    println!(
        "Multi-rank report generated under {}\nIndividual pages: rank_*/index.html",
        out_path.display()
    );

    let exec_order_summary = build_exec_order_summary(&out_path, &rank_nums, &collective_schedules);

    let diagnostics = Diagnostics {
        divergence: DivergenceFlags {
            cache: cache_seq_groups.len() > 1,
            collective: collective_seq_groups.len() > 1,
            tensor_meta: tensor_meta_groups.len() > 1,
        },
        artifacts: ArtifactFlags {
            runtime_trace: !runtime_estimations.is_empty(),
        },
        analysis: runtime_analysis,
        cache_groups: cache_divergence_groups.clone(),
        collective_groups: collective_divergence_groups.clone(),
        tensor_meta_groups: tensor_meta_divergence_groups.clone(),
        exec_order: exec_order_summary,
    };

    // Emit landing page identical to CLI
    let has_chromium_events = !all_chromium_events.is_empty();
    let show_desync_warning = compile_id_divergence
        || diagnostics.divergence.cache
        || diagnostics.divergence.collective
        || diagnostics.divergence.tensor_meta;

    let (landing_page_path, landing_html) = generate_multi_rank_html(
        &out_path,
        sorted_ranks,
        cfg,
        has_chromium_events,
        show_desync_warning,
        compile_id_divergence,
        diagnostics,
    )?;
    fs::write(&landing_page_path, landing_html)?;

    Ok(landing_page_path)
}

/// Build ExecOrderSummary from artifacts under out_path for the given ranks
pub fn build_exec_order_summary(
    out_path: &PathBuf,
    rank_nums: &[u32],
    collective_schedules: &[CollectiveSchedule],
) -> Option<ExecOrderSummary> {
    use crate::execution_order::{
        analyze_execution_order, parse_graph_execution_order, ExecOrderIssue,
    };
    use std::collections::HashSet;

    // Preload and parse compile_directory.json per rank
    let cd_by_rank: FxHashMap<u32, serde_json::Map<String, serde_json::Value>> = rank_nums
        .iter()
        .filter_map(|&rank| {
            let path = out_path
                .join(format!("rank_{rank}"))
                .join("compile_directory.json");
            std::fs::read_to_string(path)
                .ok()
                .and_then(|content| serde_json::from_str::<serde_json::Value>(&content).ok())
                .and_then(|val| match val {
                    serde_json::Value::Object(map) => Some((rank, map)),
                    _ => None,
                })
        })
        .collect();

    // Collect latest "graph_execution" artifact per rank
    let exec_orders: FxHashMap<u32, Vec<String>> = rank_nums
        .iter()
        .filter_map(|&rank| {
            let map = cd_by_rank.get(&rank)?;
            let rank_dir = out_path.join(format!("rank_{rank}"));

            // Find latest graph_execution artifact
            let best = map
                .values()
                .filter_map(|entry| entry.get("artifacts")?.as_array())
                .flat_map(|arts| arts.iter())
                .filter_map(|a| {
                    let name = a.get("name")?.as_str()?;
                    if !name.contains("graph_execution") || !name.ends_with(".json") {
                        return None;
                    }
                    Some((
                        a.get("number")?.as_u64()?,
                        a.get("url")?.as_str()?.to_string(),
                    ))
                })
                .max_by_key(|(num, _)| *num)?;

            let path = rank_dir.join(best.1);
            let order = std::fs::read_to_string(path)
                .ok()
                .and_then(|payload| parse_graph_execution_order(&payload).ok())
                .map(|order| order.into_iter().map(|s| format!("[{}]", s)).collect())?;
            Some((rank, order))
        })
        .collect();

    if exec_orders.len() < 2 {
        return None;
    }

    // Build dir -> compile_id mapping per rank
    let dir_to_compile_id_per_rank: FxHashMap<u32, FxHashMap<String, String>> = rank_nums
        .iter()
        .filter_map(|&rank| {
            let obj = cd_by_rank.get(&rank)?;
            let mapping = obj
                .iter()
                .flat_map(|(cid, entry)| {
                    entry
                        .get("artifacts")
                        .and_then(|x| x.as_array())
                        .map(|arts| {
                            arts.iter()
                                .filter_map(|a| {
                                    let url = a.get("url")?.as_str()?;
                                    let prefix = url.split_once('/')?.0;
                                    Some((prefix.to_string(), cid.to_string()))
                                })
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default()
                })
                .fold(FxHashMap::default(), |mut acc, (prefix, cid)| {
                    acc.entry(prefix).or_insert(cid);
                    acc
                });
            Some((rank, mapping))
        })
        .collect();

    // Build collective ops mapping
    let collective_by_graph: FxHashMap<(u32, String), Vec<String>> = collective_schedules
        .iter()
        .filter_map(|cs| {
            let m = dir_to_compile_id_per_rank.get(&cs.rank)?;
            let compile_id = m
                .get(&cs.graph)
                .cloned()
                .unwrap_or_else(|| cs.graph.clone());
            Some(((cs.rank, compile_id), cs.ops.clone()))
        })
        .fold(FxHashMap::default(), |mut acc, (key, ops)| {
            acc.entry(key).or_default().extend(ops);
            acc
        });

    // Build cache status mapping
    let cache_status: FxHashMap<(u32, String), String> = rank_nums
        .iter()
        .flat_map(|&rank| {
            cd_by_rank
                .get(&rank)
                .and_then(|obj| {
                    dir_to_compile_id_per_rank
                        .get(&rank)
                        .map(|dir2cid| (obj, dir2cid))
                })
                .map(|(obj, dir2cid)| {
                    let status_by_dir = obj
                        .values()
                        .filter_map(|entry| entry.get("artifacts").and_then(|x| x.as_array()))
                        .flat_map(|arts| arts.iter())
                        .filter_map(|a| {
                            let url = a.get("url")?.as_str()?;
                            let prefix = url.split_once('/')?.0.to_string();
                            let name = a.get("name")?.as_str()?;
                            let status = match () {
                                _ if name.contains("cache_miss") => ("miss", 3),
                                _ if name.contains("cache_hit") => ("hit", 2),
                                _ if name.contains("cache_bypass") => ("bypass", 1),
                                _ => return None,
                            };
                            Some((prefix, status))
                        })
                        .fold(
                            FxHashMap::default(),
                            |mut acc, (prefix, (status, priority))| {
                                acc.entry(prefix)
                                    .and_modify(|e: &mut (&str, u8)| {
                                        if priority > e.1 {
                                            *e = (status, priority);
                                        }
                                    })
                                    .or_insert((status, priority));
                                acc
                            },
                        );

                    status_by_dir
                        .into_iter()
                        .filter_map(|(dir, (st, _))| {
                            let cid = dir2cid.get(&dir)?;
                            Some(((rank, cid.clone()), st.to_string()))
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        })
        .collect();

    // Analyze and create summary
    let report = analyze_execution_order(&exec_orders, &collective_by_graph, &cache_status);

    let order_differs = report.by_index.iter().any(|row| {
        let uniq: HashSet<_> = row.by_rank.values().map(String::as_str).collect();
        uniq.len() > 1
    });

    let (sched_set, cache_set) = report.by_index.iter().fold(
        (HashSet::new(), HashSet::new()),
        |(mut sched, mut cache), row| {
            if row.issues.contains(&ExecOrderIssue::ScheduleMismatch) {
                sched.extend(row.by_rank.keys());
            }
            if row.issues.contains(&ExecOrderIssue::CacheMismatch) {
                cache.extend(row.by_rank.keys());
            }
            (sched, cache)
        },
    );

    let mut ranks_schedule: Vec<u32> = sched_set.into_iter().collect();
    let mut ranks_cache: Vec<u32> = cache_set.into_iter().collect();
    ranks_schedule.sort_unstable();
    ranks_cache.sort_unstable();

    let format_ranks = |ranks: &[u32]| {
        if ranks.is_empty() {
            String::new()
        } else {
            ranks
                .iter()
                .map(|r| format!("Rank {r}"))
                .collect::<Vec<_>>()
                .join(", ")
        }
    };

    Some(ExecOrderSummary {
        order_differs,
        has_schedule_mismatch: !ranks_schedule.is_empty(),
        has_cache_mismatch: !ranks_cache.is_empty(),
        ranks_schedule_str: format_ranks(&ranks_schedule),
        ranks_cache_str: format_ranks(&ranks_cache),
        ranks_schedule,
        ranks_cache,
    })
}

fn prepare_and_validate_graphs(
    runtime_estimations: &[GraphRuntime],
) -> Option<(
    std::collections::HashMap<u32, Vec<(&str, f64)>>,
    Vec<u32>,
    usize,
)> {
    use std::collections::HashMap;

    let mut rank_graphs: HashMap<u32, Vec<(&str, f64)>> = runtime_estimations
        .iter()
        .map(|gr| {
            (
                gr.rank,
                &gr.graph,
                gr.ops.iter().map(|op| op.estimated_runtime_ns).sum::<f64>(),
            )
        })
        .fold(HashMap::new(), |mut acc, (rank, graph, runtime)| {
            acc.entry(rank).or_default().push((graph, runtime));
            acc
        });

    // Artifacts arrive in filesystem order, which can differ across ranks.
    // Align matching graphs before comparing their runtimes by position.
    for graphs in rank_graphs.values_mut() {
        graphs.sort_by(|a, b| a.0.cmp(b.0));
    }

    let max_graphs = rank_graphs.values().map(|graphs| graphs.len()).max()?;
    let min_graphs = rank_graphs.values().map(|graphs| graphs.len()).min()?;

    if max_graphs != min_graphs {
        return None; // Different number of graphs across ranks
    }

    let mut ranks: Vec<_> = rank_graphs.keys().copied().collect();
    ranks.sort_unstable();

    Some((rank_graphs, ranks, max_graphs))
}

fn compare_graph_runtimes(
    rank_graphs: std::collections::HashMap<u32, Vec<(&str, f64)>>,
    ranks: Vec<u32>,
    max_graphs: usize,
) -> Vec<GraphAnalysis> {
    (0..max_graphs)
        .filter_map(|index| {
            let runtimes: Vec<_> = ranks
                .iter()
                .map(|&rank| {
                    rank_graphs
                        .get(&rank)
                        .and_then(|g| g.get(index))
                        .map(|(graph_id, runtime)| (rank, *graph_id, *runtime))
                })
                .collect::<Option<Vec<_>>>()?;

            let (min_runtime, max_runtime, fastest_rank, slowest_rank) = runtimes.iter().fold(
                (f64::INFINITY, f64::NEG_INFINITY, 0_u32, 0_u32),
                |(min_rt, max_rt, fast_rank, slow_rank), &(rank, _, rt)| {
                    let (new_min_rt, new_fast) = if rt <= min_rt {
                        (rt, rank)
                    } else {
                        (min_rt, fast_rank)
                    };
                    let (new_max_rt, new_slow) = if rt >= max_rt {
                        (rt, rank)
                    } else {
                        (max_rt, slow_rank)
                    };
                    (new_min_rt, new_max_rt, new_fast, new_slow)
                },
            );

            let delta_ns = max_runtime - min_runtime;

            Some(GraphAnalysis {
                graph_index: index,
                graph_id: runtimes[0].1.to_string(),
                delta_ms: (delta_ns / 1e6 * 1000.0).round() / 1000.0,
                rank_details: vec![
                    RuntimeRankDetail {
                        rank: fastest_rank,
                        runtime_ms: (min_runtime / 1e6 * 1000.0).round() / 1000.0,
                    },
                    RuntimeRankDetail {
                        rank: slowest_rank,
                        runtime_ms: (max_runtime / 1e6 * 1000.0).round() / 1000.0,
                    },
                ],
            })
        })
        .collect()
}

pub fn analyze_graph_runtime_deltas(
    runtime_estimations: &[GraphRuntime],
) -> Option<RuntimeAnalysis> {
    let Some((rank_graphs, ranks, max_graphs)) = prepare_and_validate_graphs(runtime_estimations)
    else {
        return Some(RuntimeAnalysis {
            graphs: vec![],
            has_mismatched_graph_counts: true,
        });
    };

    let mut graphs = compare_graph_runtimes(rank_graphs, ranks, max_graphs);
    graphs.sort_by(|a, b| a.graph_id.cmp(&b.graph_id));

    Some(RuntimeAnalysis {
        graphs,
        has_mismatched_graph_counts: false,
    })
}

/// Converts node-based mappings to line number-based mappings for visualization.
///
/// This function processes node mappings and converts them to line number mappings
/// that can be used to highlight corresponding lines across different views.
/// It handles pre-grad graph, post-grad graph, and generated code files.
fn convert_node_mappings_to_line_numbers(
    node_mappings_content: &str,
    pre_grad_graph_content: &str,
    post_grad_graph_content: &str,
    output_code_content: &str,
    aot_code_content: &str,
) -> serde_json::Value {
    // Parse the node mappings JSON
    let node_mappings: serde_json::Value = match serde_json::from_str(node_mappings_content) {
        Ok(mappings) => mappings,
        Err(_) => return serde_json::json!({}),
    };

    let version = node_mappings
        .get("version")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as i64;

    // Helper function to check if a line is valid (not empty and doesn't start with comment)
    fn valid_line(line: &str, symbol: &str) -> bool {
        let stripped = line.trim();
        !stripped.is_empty() && !stripped.starts_with(symbol)
    }

    // Helper function to extract node name from a line
    fn extract_node_name(line: &str) -> Option<String> {
        let trimmed = line.trim();
        if valid_line(trimmed, "#") {
            // Split on '=' and take everything before it
            let before_equals = trimmed.split('=').next()?;
            // Split on ':' and take everything before it
            let node_name = before_equals.split(':').next()?.trim();
            if !node_name.is_empty() {
                return Some(node_name.to_string());
            }
        }
        None
    }

    // Helper function to build node-to-line lookup map from graph content
    fn build_node_to_lines_map(content: &str) -> std::collections::HashMap<String, usize> {
        let mut node_to_lines = std::collections::HashMap::new();
        for (i, line) in content.lines().enumerate() {
            if let Some(node_name) = extract_node_name(line) {
                node_to_lines.insert(node_name, i + 1); // 1-based line numbers
            }
        }
        node_to_lines
    }

    // Helper function to build Python kernel-to-lines lookup map
    fn build_python_kernel_to_lines_map(
        content: &str,
        kernel_names: &[&str],
        _version: i64,
    ) -> std::collections::HashMap<String, Vec<usize>> {
        let content = content
            .lines()
            .skip_while(|line| line.is_empty())
            .collect::<Vec<&str>>()
            .join("\n");
        let mut kernel_to_lines = std::collections::HashMap::new();

        // Find the line number of "def call(args)" - allowing for whitespace between tokens
        let run_impl_line = content
            .lines()
            .position(|line| {
                line.contains("def") && line.contains("call") && line.contains("(args)")
            })
            .unwrap_or(0);
        let first_line_number = content
            .lines()
            .position(|line| line.contains("# AOT ID:"))
            .unwrap_or(0);

        // For each kernel name (e.g. triton_poi_fused_mul_1:2):
        // - Extract pure_kernel_name (triton_poi_fused_mul_1) before the ':'
        // - If kernel name found: map to next line containing pure_kernel_name
        // - If kernel_name not found: map to all lines with pure_kernel_name
        for kernel_name in kernel_names {
            // Get pure kernel name before ':' if it exists
            let pure_kernel_name = if let Some(idx) = kernel_name.find(':') {
                &kernel_name[..idx]
            } else {
                kernel_name
            };

            let mut found = false;
            // If kernel_name contains a debug handle and we found it, we can stop after first match
            if kernel_name.contains(':') {
                for (i, line) in content.lines().enumerate().skip(run_impl_line) {
                    if line.contains(kernel_name) {
                        // Found kernel name, look for next line with pure_kernel_name
                        for (j, next_line) in content.lines().enumerate().skip(i + 1) {
                            if next_line.contains(pure_kernel_name) {
                                kernel_to_lines
                                    .entry(kernel_name.to_string())
                                    .or_insert_with(Vec::new)
                                    .push(j + 1 - first_line_number);
                                found = true;
                                break;
                            }
                        }
                        break;
                    }
                }
            }

            // If exact kernel name not found, map all lines with pure kernel name
            if !found {
                for (i, line) in content.lines().enumerate().skip(run_impl_line) {
                    if line.contains(pure_kernel_name) {
                        kernel_to_lines
                            .entry(kernel_name.to_string())
                            .or_insert_with(Vec::new)
                            .push(i + 1 - first_line_number);
                    }
                }
            }
        }
        kernel_to_lines
    }

    // Helper function to build C++ kernel-to-lines lookup map
    // We only consider lines after "::run_impl(" and skip the empty lines at the beginning when computing line numbers
    fn build_cpp_kernel_to_lines_map(
        content: &str,
        kernel_names: &[&str],
        _version: i64,
    ) -> std::collections::HashMap<String, Vec<usize>> {
        // remove empty lines at the beginning and end of the content
        // We need to do this because empty lines are ignored in html <pre> tags
        let content = content
            .lines()
            .skip_while(|line| line.is_empty())
            .collect::<Vec<&str>>()
            .join("\n");
        let mut kernel_to_lines = std::collections::HashMap::new();

        // Find the line number of "::run_impl("
        let run_impl_line = content
            .lines()
            .position(|line| line.contains("::run_impl("))
            .unwrap_or(0);

        // For each kernel name (e.g. triton_poi_fused_mul_1:2):
        // - Extract pure_kernel_name (triton_poi_fused_mul_1) before the ':'
        // - If kernel name found: map to next line containing pure_kernel_name
        // - If kernel_name not found: map to all lines with pure_kernel_name
        for kernel_name in kernel_names {
            // Get pure kernel name before ':' if it exists
            let pure_kernel_name = if let Some(idx) = kernel_name.rfind(':') {
                &kernel_name[..idx]
            } else {
                kernel_name
            };

            let mut found = false;
            if kernel_name.contains(':') {
                for (i, line) in content.lines().enumerate().skip(run_impl_line) {
                    if valid_line(line, "def")
                        && valid_line(line, "static inline void")
                        && line.contains(kernel_name)
                    {
                        // Found exact kernel name - map to next matching line
                        let next_line = content
                            .lines()
                            .skip(i + 1)
                            // Filter out variable declarations like "int64_t kernel_xnumel_1 = value;" but keep function calls
                            .position(|l| {
                                l.contains(pure_kernel_name)
                                    && !l.contains("_xnumel = ")
                                    && !Regex::new(r"^\s*\w+\s+.*_xnumel(?:_\d+)?\s*=")
                                        .unwrap()
                                        .is_match(l)
                            })
                            .map(|pos| i + pos + 2);

                        if let Some(line_num) = next_line {
                            kernel_to_lines
                                .entry(kernel_name.to_string())
                                .or_insert_with(Vec::new)
                                .push(line_num);
                            found = true;
                            break;
                        }
                    }
                }
            }
            if !found {
                for (i, line) in content.lines().enumerate().skip(run_impl_line) {
                    if line.contains(pure_kernel_name) {
                        kernel_to_lines
                            .entry(kernel_name.to_string())
                            .or_insert_with(Vec::new)
                            .push(i + 1);
                    }
                }
            }
        }
        kernel_to_lines
    }

    // Helper function to process mappings from source to target
    fn process_mappings<F>(
        source_mappings: &serde_json::Map<String, serde_json::Value>,
        source_lookup: &std::collections::HashMap<String, usize>,
        _target_lookup: &std::collections::HashMap<String, usize>,
        target_line_processor: F,
    ) -> std::collections::HashMap<usize, Vec<usize>>
    where
        F: Fn(&str) -> Option<usize>,
    {
        let mut result = std::collections::HashMap::new();

        for (source_node, target_nodes) in source_mappings {
            if let Some(source_line) = source_lookup.get(source_node) {
                let mut target_lines = Vec::new();
                if let Some(target_nodes_array) = target_nodes.as_array() {
                    for target_node in target_nodes_array {
                        if let Some(target_node_str) = target_node.as_str() {
                            if let Some(target_line) = target_line_processor(target_node_str) {
                                target_lines.push(target_line);
                            }
                        }
                    }
                }
                if !target_lines.is_empty() {
                    result.insert(*source_line, target_lines);
                }
            }
        }
        result
    }

    // Helper function to process kernel-to-post mappings
    fn process_kernel_to_post_mappings(
        kernel_mappings: &serde_json::Map<String, serde_json::Value>,
        kernel_lookup: &std::collections::HashMap<String, Vec<usize>>,
        post_lookup: &std::collections::HashMap<String, usize>,
    ) -> std::collections::HashMap<usize, Vec<usize>> {
        let mut result = std::collections::HashMap::new();

        for (kernel_name, post_nodes) in kernel_mappings {
            if let Some(kernel_lines) = kernel_lookup.get(kernel_name) {
                for kernel_line in kernel_lines {
                    let mut target_lines = Vec::new();
                    if let Some(post_nodes_array) = post_nodes.as_array() {
                        for post_node in post_nodes_array {
                            if let Some(post_node_str) = post_node.as_str() {
                                if let Some(post_line) = post_lookup.get(post_node_str) {
                                    target_lines.push(*post_line);
                                }
                            }
                        }
                    }
                    if !target_lines.is_empty() {
                        result.insert(*kernel_line, target_lines);
                    }
                }
            }
        }
        result
    }

    // Helper function to process post-to-kernel mappings
    fn process_post_to_kernel_mappings(
        post_mappings: &serde_json::Map<String, serde_json::Value>,
        post_lookup: &std::collections::HashMap<String, usize>,
        kernel_lookup: &std::collections::HashMap<String, Vec<usize>>,
    ) -> std::collections::HashMap<usize, Vec<usize>> {
        let mut result = std::collections::HashMap::new();

        for (post_node, kernel_names) in post_mappings {
            if let Some(post_line) = post_lookup.get(post_node) {
                let mut target_lines = Vec::new();
                if let Some(kernel_names_array) = kernel_names.as_array() {
                    for kernel_name in kernel_names_array {
                        if let Some(kernel_name_str) = kernel_name.as_str() {
                            if let Some(kernel_lines) = kernel_lookup.get(kernel_name_str) {
                                target_lines.extend(kernel_lines);
                            }
                        }
                    }
                }
                if !target_lines.is_empty() {
                    result.insert(*post_line, target_lines);
                }
            }
        }
        result
    }

    // Helper function to convert HashMap to JSON Map
    fn hashmap_to_json_map(
        map: std::collections::HashMap<usize, Vec<usize>>,
    ) -> serde_json::Map<String, serde_json::Value> {
        map.into_iter()
            .map(|(k, v)| (k.to_string(), serde_json::json!(v)))
            .collect()
    }

    let kernel_names: Vec<&str> = node_mappings
        .get("cppCodeToPost")
        .and_then(|v| v.as_object())
        .map(|obj| obj.keys().map(|k| k.as_str()).collect())
        .unwrap_or_default();

    // Build lookup maps
    let pre_grad_node_to_lines = build_node_to_lines_map(pre_grad_graph_content);
    let post_grad_node_to_lines = build_node_to_lines_map(post_grad_graph_content);
    let py_kernel_to_lines =
        build_python_kernel_to_lines_map(output_code_content, &kernel_names, version);
    let cpp_code_to_lines = build_cpp_kernel_to_lines_map(aot_code_content, &kernel_names, version);

    // Process all mappings using helper functions
    let line_pre_to_post =
        if let Some(pre_to_post) = node_mappings.get("preToPost").and_then(|v| v.as_object()) {
            process_mappings(
                pre_to_post,
                &pre_grad_node_to_lines,
                &post_grad_node_to_lines,
                |node_name| post_grad_node_to_lines.get(node_name).copied(),
            )
        } else {
            std::collections::HashMap::new()
        };

    let line_post_to_pre =
        if let Some(post_to_pre) = node_mappings.get("postToPre").and_then(|v| v.as_object()) {
            process_mappings(
                post_to_pre,
                &post_grad_node_to_lines,
                &pre_grad_node_to_lines,
                |node_name| pre_grad_node_to_lines.get(node_name).copied(),
            )
        } else {
            std::collections::HashMap::new()
        };

    let line_cpp_code_to_post = if let Some(cpp_code_to_post) = node_mappings
        .get("cppCodeToPost")
        .and_then(|v| v.as_object())
    {
        process_kernel_to_post_mappings(
            cpp_code_to_post,
            &cpp_code_to_lines,
            &post_grad_node_to_lines,
        )
    } else {
        std::collections::HashMap::new()
    };

    let line_post_to_cpp_code = if let Some(post_to_cpp_code) = node_mappings
        .get("postToCppCode")
        .and_then(|v| v.as_object())
    {
        process_post_to_kernel_mappings(
            post_to_cpp_code,
            &post_grad_node_to_lines,
            &cpp_code_to_lines,
        )
    } else {
        std::collections::HashMap::new()
    };

    let line_py_code_to_post = if let Some(cpp_code_to_post) = node_mappings
        .get("cppCodeToPost")
        .and_then(|v| v.as_object())
    {
        process_kernel_to_post_mappings(
            cpp_code_to_post,
            &py_kernel_to_lines,
            &post_grad_node_to_lines,
        )
    } else {
        std::collections::HashMap::new()
    };

    let line_post_to_py_code = if let Some(post_to_cpp_code) = node_mappings
        .get("postToCppCode")
        .and_then(|v| v.as_object())
    {
        process_post_to_kernel_mappings(
            post_to_cpp_code,
            &post_grad_node_to_lines,
            &py_kernel_to_lines,
        )
    } else {
        std::collections::HashMap::new()
    };

    // Convert all HashMaps to JSON objects
    serde_json::json!({
        "preToPost": hashmap_to_json_map(line_pre_to_post),
        "postToPre": hashmap_to_json_map(line_post_to_pre),
        "pyCodeToPost": hashmap_to_json_map(line_py_code_to_post),
        "postToPyCode": hashmap_to_json_map(line_post_to_py_code),
        "cppCodeToPost": hashmap_to_json_map(line_cpp_code_to_post),
        "postToCppCode": hashmap_to_json_map(line_post_to_cpp_code)
    })
}

pub mod execution_order {
    use fxhash::FxHashMap;

    /// Issue types detected at a given execution index across ranks
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ExecOrderIssue {
        ScheduleMismatch,
        CacheMismatch,
    }

    /// One row in the execution-order report
    #[derive(Debug, Clone)]
    pub struct ExecOrderIndexRow {
        /// Zero-based index into per-rank execution orders
        pub index: usize,
        /// Mapping: rank -> compile_id (compile directory name)
        pub by_rank: FxHashMap<u32, String>,
        /// Issues found for this index
        pub issues: Vec<ExecOrderIssue>,
    }

    /// Final report for execution-order diagnostics
    #[derive(Debug, Clone, Default)]
    pub struct ExecOrderReport {
        pub by_index: Vec<ExecOrderIndexRow>,
    }

    /// Analyze per-rank execution orders, aligning entries by index and flagging issues
    /// using provided per-graph properties.
    pub fn analyze_execution_order(
        exec_orders: &FxHashMap<u32, Vec<String>>,
        collective_schedule_by_graph: &FxHashMap<(u32, String), Vec<String>>,
        cache_status: &FxHashMap<(u32, String), String>,
    ) -> ExecOrderReport {
        // Determine max length across ranks (N)
        let max_len = exec_orders.values().map(|v| v.len()).max().unwrap_or(0);

        if max_len == 0 || exec_orders.is_empty() {
            return ExecOrderReport::default();
        }

        // Memoize property lookups per (rank, compile_id)
        let mut sched_memo: FxHashMap<(u32, String), Option<Vec<String>>> = FxHashMap::default();
        let mut cache_memo: FxHashMap<(u32, String), Option<String>> = FxHashMap::default();

        let mut rows: Vec<ExecOrderIndexRow> = Vec::with_capacity(max_len);

        for idx in 0..max_len {
            // Gather present ranks and their compile_ids at this index
            let by_rank: FxHashMap<u32, String> = exec_orders
                .iter()
                .filter_map(|(&rank, order)| order.get(idx).cloned().map(|cid| (rank, cid)))
                .collect();

            if by_rank.is_empty() {
                continue;
            }

            // Evaluate issues among present ranks
            let mut issues: Vec<ExecOrderIssue> = Vec::new();

            // Schedule mismatch: compare collective schedules regardless of compile_id
            {
                let schedules: Vec<Vec<String>> = by_rank
                    .iter()
                    .filter_map(|(&rank, cid)| {
                        let key = (rank, cid.clone());
                        let entry = sched_memo
                            .entry((rank, cid.clone()))
                            .or_insert_with(|| collective_schedule_by_graph.get(&key).cloned());
                        entry.clone()
                    })
                    .collect();
                if schedules.len() >= 2 && schedules[1..].iter().any(|s| s != &schedules[0]) {
                    issues.push(ExecOrderIssue::ScheduleMismatch);
                }
            }

            // Cache mismatch: compare cache statuses regardless of compile_id
            {
                let statuses: Vec<String> = by_rank
                    .iter()
                    .filter_map(|(&rank, cid)| {
                        let key = (rank, cid.clone());
                        let entry = cache_memo
                            .entry((rank, cid.clone()))
                            .or_insert_with(|| cache_status.get(&key).cloned());
                        entry.clone().filter(|s| !s.is_empty())
                    })
                    .collect();
                if statuses.len() >= 2 && statuses[1..].iter().any(|s| s != &statuses[0]) {
                    issues.push(ExecOrderIssue::CacheMismatch);
                }
            }

            rows.push(ExecOrderIndexRow {
                index: idx,
                by_rank,
                issues,
            });
        }

        ExecOrderReport { by_index: rows }
    }

    pub fn parse_graph_execution_order(payload: &str) -> anyhow::Result<Vec<String>> {
        let value: serde_json::Value = serde_json::from_str(payload)?;
        let arr = value
            .get("graph_execution_order")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("missing graph_execution_order array"))?;

        let mut out = Vec::with_capacity(arr.len());
        for item in arr {
            match item {
                serde_json::Value::String(s) => out.push(s.clone()),
                serde_json::Value::Object(map) => {
                    if let Some(s) = map.get("compile_id").and_then(|v| v.as_str()) {
                        out.push(s.to_string());
                    }
                }
                _ => {}
            }
        }
        Ok(out)
    }
}
