use core::hash::BuildHasherDefault;
use fxhash::{FxHashMap, FxHasher};
use html_escape::encode_text;
use indexmap::IndexMap;
use regex::Regex;
use serde_json::Value;

use std::fmt::{self, Display, Write};
use std::path::PathBuf;

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

// Main function returns a list of files to save
pub type ParseOutput = Vec<(PathBuf, String)>;
pub type CompilationMetricsIndex = FxIndexMap<Option<CompileId>, Vec<CompilationMetricsMetadata>>;
pub type StackIndex = FxHashMap<Option<CompileId>, StackSummary>; // NB: attempt is always 0 here
pub type SymbolicShapeSpecializationIndex =
    FxHashMap<Option<CompileId>, Vec<SymbolicShapeSpecializationMetadata>>;

pub type FxIndexMap<K, V> = IndexMap<K, V, BuildHasherDefault<FxHasher>>;

pub fn extract_eval_with_key_id(filename: &str) -> Option<u64> {
    let re = Regex::new(r"<eval_with_key>\.([0-9]+)").unwrap();
    re.captures(filename)
        .and_then(|caps| caps.get(1))
        .and_then(|m| m.as_str().parse::<u64>().ok())
}

pub static INTERN_TABLE: Lazy<Mutex<FxHashMap<u32, String>>> =
    Lazy::new(|| Mutex::new(FxHashMap::default()));

#[derive(Default)]
pub struct StackTrieNode {
    terminal: Vec<Option<CompileId>>,
    // Ordered map so that when we print we roughly print in chronological order
    children: FxIndexMap<FrameSummary, StackTrieNode>,
}

impl StackTrieNode {
    pub fn insert(&mut self, mut stack: StackSummary, compile_id: Option<CompileId>) {
        let mut cur = self;
        for frame in stack.drain(..) {
            cur = cur.children.entry(frame).or_default();
        }
        cur.terminal.push(compile_id);
    }

    pub fn insert_no_terminal(&mut self, mut stack: StackSummary) {
        let mut cur = self;
        for frame in stack.drain(..) {
            cur = cur.children.entry(frame).or_default();
        }
    }

    pub fn is_empty(&self) -> bool {
        return self.children.is_empty() && self.terminal.is_empty();
    }

    pub fn fmt(
        &self,
        metrics_index: Option<&CompilationMetricsIndex>,
    ) -> Result<String, fmt::Error> {
        let mut f = String::new();
        write!(f, "<div class='stack-trie'>")?;
        write!(f, "<ul>")?;
        self.fmt_inner(&mut f, metrics_index)?;
        write!(f, "</ul>")?;
        write!(f, "</div>")?;
        Ok(f)
    }

    pub fn fmt_inner(
        &self,
        f: &mut String,
        mb_metrics_index: Option<&CompilationMetricsIndex>,
    ) -> fmt::Result {
        for (frame, node) in self.children.iter() {
            let mut star = String::new();
            for t in &node.terminal {
                if let Some(c) = t {
                    let ok_class = mb_metrics_index.map_or("status-missing", |metrics_index| {
                        metrics_index.get(t).map_or("status-missing", |m| {
                            if m.iter().any(|n| n.fail_type.is_some()) {
                                "status-error"
                            } else if m.iter().any(|n| n.graph_op_count.unwrap_or(0) == 0) {
                                "status-empty"
                            } else if m.iter().any(|n| {
                                !n.restart_reasons.as_ref().map_or(false, |o| o.is_empty())
                            }) {
                                "status-break"
                            } else {
                                "status-ok"
                            }
                        })
                    });
                    write!(
                        star,
                        "<a href='#{cid}' class='{ok_class}'>{cid}</a> ",
                        cid = c,
                        ok_class = ok_class
                    )?;
                } else {
                    write!(star, "(unknown) ")?;
                }
            }

            if self.children.len() > 1 {
                // If the node has multiple children, increase the indent and print a hyphen
                writeln!(
                    f,
                    "<li><span onclick='toggleList(this)' class='marker'></span>{star}",
                    star = star
                )?;
                writeln!(f, "{}<ul>", frame)?;
                node.fmt_inner(f, mb_metrics_index)?;
                write!(f, "</ul></li>")?;
            } else {
                // If the node has only one child, don't increase the indent and don't print a hyphen
                writeln!(f, "<li>{star}{}</li>", frame, star = star)?;
                node.fmt_inner(f, mb_metrics_index)?;
            }
        }
        Ok(())
    }
}

#[derive(Eq, PartialEq, Hash, Deserialize, Serialize, Debug, Clone)]
pub struct CompileId {
    pub frame_id: u32,
    pub frame_compile_id: u32,
    pub attempt: u32,
}

impl fmt::Display for CompileId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}/{}", self.frame_id, self.frame_compile_id)?;
        if self.attempt != 0 {
            write!(f, "_{}", self.attempt)?;
        }
        write!(f, "]")
    }
}

#[derive(Default, Debug)]
pub struct Stats {
    pub ok: u64,
    pub other_rank: u64,
    pub fail_glog: u64,
    pub fail_json: u64,
    pub fail_payload_md5: u64,
    pub fail_dynamo_guards_json: u64,
    pub fail_parser: u64,
    pub unknown: u64,
}

#[derive(Debug, Hash, Eq, PartialEq, Deserialize, Serialize, Clone)]
pub struct FrameSummary {
    pub filename: u32,
    pub line: i32,
    pub name: String,
    pub uninterned_filename: Option<String>,
}

pub fn simplify_filename<'a>(filename: &'a str) -> &'a str {
    let parts: Vec<&'a str> = filename.split("#link-tree/").collect();
    if parts.len() > 1 {
        return parts[1];
    }
    let re = Regex::new(r"[^/]+-seed-nspid[^/]+/").unwrap();
    if let Some(captures) = re.captures(filename) {
        if let Some(capture) = captures.get(0) {
            return &filename[capture.end()..];
        }
    }
    return filename;
}

pub fn unintern_str(interned_str: u32) -> String {
    let intern_table = INTERN_TABLE.lock().unwrap();
    let filename = intern_table
        .get(&interned_str)
        .map_or("(unknown)", |s| s.as_str());
    return filename.to_string();
}

impl fmt::Display for FrameSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let intern_table = INTERN_TABLE.lock().unwrap();
        let filename = if let Some(f) = &self.uninterned_filename {
            f.as_str()
        } else {
            intern_table
                .get(&self.filename)
                .map_or("(unknown)", |s| s.as_str())
        };
        if let Some(fx_id) = extract_eval_with_key_id(filename) {
            write!(
                f,
                "<a href='dump_file/eval_with_key_{fx_id}.html#L{line}'>{filename}:{line}</a> in {name}",
                fx_id = fx_id,
                filename = encode_text(simplify_filename(filename)),
                line = self.line,
                name = encode_text(&self.name)
            )?;
        } else {
            write!(
                f,
                "{}:{} in {}",
                encode_text(simplify_filename(filename)),
                self.line,
                encode_text(&self.name)
            )?;
        }
        Ok(())
    }
}

pub type StackSummary = Vec<FrameSummary>;

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum SymInt {
    Int(i64),
    Symbol(String),
}

impl Default for SymInt {
    fn default() -> Self {
        SymInt::Int(0)
    }
}

fn default_layout() -> String {
    "torch.strided".to_string()
}

#[derive(Debug, Deserialize)]
pub struct OptimizeDdpSplitChildMetadata {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct EmptyMetadata {}

#[derive(Debug, Deserialize)]
pub struct GraphDumpMetadata {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct DynamoOutputGraphMetadata {
    _sizes: Option<FxHashMap<String, Vec<SymInt>>>,
}

#[derive(Debug, Deserialize)]
pub struct DynamoStartMetadata {
    pub stack: Option<StackSummary>,
}

#[derive(Debug, Deserialize)]
pub struct InductorOutputCodeMetadata {
    pub filename: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
pub struct LinkMetadata {
    pub name: String,
    pub url: String,
}

#[derive(Debug, Deserialize)]
pub struct ArtifactMetadata {
    pub name: String,
    pub encoding: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CompilationMetricsMetadata {
    // Other information like frame_key are already in envelope
    pub co_name: Option<String>,
    pub co_filename: Option<String>,
    pub co_firstlineno: Option<i32>,
    pub cache_size: Option<u64>,
    pub accumulated_cache_size: Option<u64>,
    pub guard_count: Option<u64>,
    pub shape_env_guard_count: Option<u64>,
    pub graph_op_count: Option<u64>,
    pub graph_node_count: Option<u64>,
    pub graph_input_count: Option<u64>,
    pub start_time: Option<f64>,
    pub entire_frame_compile_time_s: Option<f64>,
    pub backend_compile_time_s: Option<f64>,
    pub inductor_compile_time_s: Option<f64>,
    pub code_gen_time_s: Option<f64>,
    pub fail_type: Option<String>,
    pub fail_reason: Option<String>,
    pub fail_user_frame_filename: Option<String>,
    pub fail_user_frame_lineno: Option<u32>,
    pub non_compliant_ops: Option<Vec<String>>,
    pub compliant_custom_ops: Option<Vec<String>>,
    pub restart_reasons: Option<Vec<String>>,
    pub dynamo_time_before_restart_s: Option<f64>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct BwdCompilationMetricsMetadata {
    pub inductor_compile_time_s: Option<f64>,
    pub code_gen_time_s: Option<f64>,
    pub fail_type: Option<String>,
    pub fail_reason: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AOTAutogradBackwardCompilationMetricsMetadata {
    pub start_time: Option<f64>,
    pub elapsed_time: Option<f64>, // technically redundant with envelope
    pub fail_type: Option<String>,
    pub fail_reason: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SymbolicShapeSpecializationMetadata {
    pub symbol: Option<String>,
    pub sources: Option<Vec<String>>,
    pub value: Option<String>,
    pub reason: Option<String>,
    pub stack: Option<StackSummary>,
    pub user_stack: Option<StackSummary>,
}

#[derive(Debug, Serialize)]
pub struct BwdCompilationMetricsContext<'e> {
    pub m: &'e BwdCompilationMetricsMetadata,
    pub css: &'static str,
    pub compile_id: String,
}

#[derive(Debug, Serialize)]
pub struct AOTAutogradBackwardCompilationMetricsContext<'e> {
    pub m: &'e AOTAutogradBackwardCompilationMetricsMetadata,
    pub css: &'static str,
    pub compile_id: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct OutputFile {
    pub url: String,
    pub name: String,
    pub number: i32,
    pub suffix: String,
}

#[derive(Debug, Serialize)]
pub struct CompilationMetricsContext<'e> {
    pub m: &'e CompilationMetricsMetadata,
    pub css: &'static str,
    pub compile_id: String,
    pub stack_html: String,
    pub symbolic_shape_specializations: Vec<SymbolicShapeSpecializationContext>,
    pub output_files: &'e Vec<OutputFile>,
    pub compile_id_dir: &'e PathBuf,
    pub mini_stack_html: String,
}

#[derive(Debug, Serialize)]
pub enum FailureReason {
    Failure((String, String, String, u32)), // (failure type, failure reason, user frame filename, user frame lineno)
    Restart(String),                        // restart reason
}
impl Display for FailureReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FailureReason::Failure((
                failure_type,
                failure_reason,
                user_frame_filename,
                user_frame_lineno,
            )) => {
                let failure_type = encode_text(failure_type);
                let failure_reason = encode_text(failure_reason);
                let user_frame_filename = encode_text(user_frame_filename);
                write!(
                    f,
                    "<td><pre>{failure_type}</pre></td>
                           <td><pre>{failure_reason}</pre></td>
                           <td><pre>{user_frame_filename}:{user_frame_lineno}</pre></td>
                          "
                )
            }
            FailureReason::Restart(restart_reason) => write!(
                f,
                r#"<td> RestartAnalysis </td><td><pre>{restart_reason}</pre></td><td>Not availble for restarts(yet)!</td>"#
            ),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct RestartsAndFailuresContext {
    // Serialized versions of (CompileId, FailureReason)
    pub failures: Vec<(String, String)>,
    pub css: &'static str,
}

#[derive(Debug)]
pub enum Metadata<'e> {
    Empty(&'e EmptyMetadata),
    Link(&'e LinkMetadata),
    GraphDump(&'e GraphDumpMetadata),
    DynamoOutputGraph(&'e DynamoOutputGraphMetadata),
    #[allow(dead_code)]
    DynamoStart(&'e DynamoStartMetadata),
    InductorOutputCode(&'e InductorOutputCodeMetadata),
    OptimizeDdpSplitChild(&'e OptimizeDdpSplitChildMetadata),
    CompilationMetrics(&'e CompilationMetricsMetadata),
    AOTAutogradBackwardCompilationMetrics(&'e AOTAutogradBackwardCompilationMetricsMetadata),
    BwdCompilationMetrics(&'e BwdCompilationMetricsMetadata),
    Artifact(&'e ArtifactMetadata),
    DumpFile(&'e DumpFileMetadata),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DumpFileMetadata {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct Envelope {
    pub rank: Option<u32>,
    #[serde(flatten)]
    pub compile_id: Option<CompileId>,
    #[serde(default)]
    pub has_payload: Option<String>,
    pub stack: Option<StackSummary>,
    // externally tagged union, one field per log type we recognize
    pub dynamo_start: Option<DynamoStartMetadata>,
    pub str: Option<(String, u32)>,
    pub dynamo_output_graph: Option<DynamoOutputGraphMetadata>,
    pub optimize_ddp_split_graph: Option<EmptyMetadata>,
    pub optimize_ddp_split_child: Option<OptimizeDdpSplitChildMetadata>,
    pub compiled_autograd_graph: Option<EmptyMetadata>,
    pub dynamo_guards: Option<EmptyMetadata>,
    pub aot_forward_graph: Option<EmptyMetadata>,
    pub aot_backward_graph: Option<EmptyMetadata>,
    pub aot_joint_graph: Option<EmptyMetadata>,
    pub inductor_post_grad_graph: Option<EmptyMetadata>,
    pub dynamo_cpp_guards_str: Option<EmptyMetadata>,
    pub inductor_output_code: Option<InductorOutputCodeMetadata>,
    pub compilation_metrics: Option<CompilationMetricsMetadata>,
    pub bwd_compilation_metrics: Option<BwdCompilationMetricsMetadata>,
    pub aot_autograd_backward_compilation_metrics:
        Option<AOTAutogradBackwardCompilationMetricsMetadata>,
    pub graph_dump: Option<GraphDumpMetadata>,
    pub link: Option<LinkMetadata>,
    pub symbolic_shape_specialization: Option<SymbolicShapeSpecializationMetadata>,
    pub artifact: Option<ArtifactMetadata>,
    pub describe_storage: Option<StorageDesc>,
    pub describe_tensor: Option<TensorDesc>,
    pub describe_source: Option<SourceDesc>,
    pub dump_file: Option<DumpFileMetadata>,
    pub chromium_event: Option<EmptyMetadata>,
    #[serde(flatten)]
    pub _other: FxHashMap<String, Value>,
}

type MetaTensorId = u64;
type MetaStorageId = u64;

#[derive(Debug, Deserialize, Serialize)]
pub struct TensorDesc {
    id: MetaTensorId,
    describer_id: u64,
    ndim: u64,
    dtype: String,
    device: String,
    size: Vec<SymInt>,
    dynamo_dynamic_indices: Option<Vec<u64>>,
    // TODO: Make layout an enum
    #[serde(default = "default_layout")]
    layout: String,
    #[serde(default)]
    is_inference: bool,
    #[serde(default)]
    is_leaf: bool,
    #[serde(default)]
    requires_grad: bool,
    #[serde(default)]
    is_sparse: bool,
    #[serde(default)]
    is_mkldnn: bool,
    #[serde(default)]
    is_functorch_wrapped: bool,
    #[serde(default)]
    is_batchedtensor: bool,
    #[serde(default)]
    is_legacy_batchedtensor: bool,
    #[serde(default)]
    is_gradtrackingtensor: bool,
    #[serde(default)]
    is_view: bool,
    #[serde(default)]
    is_nested: bool,
    #[serde(default)]
    is_traceable_wrapper_subclass: bool,
    #[serde(default)]
    is_functional: bool,
    #[serde(default)]
    is_conj: bool,
    #[serde(default)]
    is_neg: bool,
    #[serde(default)]
    is_parameter: bool,
    stride: Option<Vec<SymInt>>,
    #[serde(default)]
    storage_offset: SymInt,
    storage: Option<MetaStorageId>,
    sparse_dim: Option<u64>,
    dense_dim: Option<u64>,
    is_coalesced: Option<bool>,
    crow_indices: Option<MetaTensorId>,
    col_indices: Option<MetaTensorId>,
    ccol_indices: Option<MetaTensorId>,
    row_indices: Option<MetaTensorId>,
    values: Option<MetaTensorId>,
    unwrapped: Option<MetaTensorId>,
    bdim: Option<u64>,
    base: Option<MetaTensorId>,
    attrs: Option<FxHashMap<String, MetaTensorId>>,
    creation_meta: Option<String>,
    grad: Option<MetaTensorId>,
    #[serde(flatten)]
    pub _other: FxHashMap<String, Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StorageDesc {
    id: MetaStorageId,
    describer_id: u64,
    size: u64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SourceDesc {
    describer_id: u64,
    id: MetaTensorId,
    source: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DynamoGuard {
    pub code: String,
    pub stack: Option<StackSummary>,
    pub user_stack: Option<StackSummary>,
}

#[derive(Debug, Serialize)]
pub struct DynamoGuardsContext {
    pub guards: Vec<DynamoGuard>,
}

#[derive(Debug, Serialize)]
pub struct IndexContext {
    pub css: &'static str,
    pub javascript: &'static str,
    pub directory: Vec<(String, Vec<OutputFile>)>,
    pub stack_trie_html: String,
    pub unknown_stack_trie_html: String,
    pub has_unknown_stack_trie: bool,
    pub num_breaks: usize,
    pub custom_header_html: String,
    pub has_chromium_events: bool,
}

#[derive(Debug, Serialize)]
pub struct SymbolicShapeSpecializationContext {
    pub symbol: String,
    pub sources: Vec<String>,
    pub value: String,
    pub user_stack_html: String,
    pub stack_html: String,
}
