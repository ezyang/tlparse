use core::hash::BuildHasherDefault;
use fxhash::{FxHashMap, FxHasher};
use html_escape::encode_text;
use indexmap::IndexMap;
use regex::Regex;

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
                    "<li><span onclick='toggleList(this)' class='marker'></span>{star}{}<ul>",
                    frame,
                    star = star
                )?;
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
}

#[derive(Debug, Hash, Eq, PartialEq, Deserialize, Serialize, Clone)]
pub struct FrameSummary {
    pub filename: u32,
    pub line: i32,
    pub name: String,
}

pub fn simplify_filename<'a>(filename: &'a str) -> &'a str {
    let parts: Vec<&'a str> = filename.split("#link-tree/").collect();
    if parts.len() > 1 {
        return parts[1];
    }
    let re = Regex::new(r"\d+e\d+-seed-nspid\d+_cgpid\d+-ns-\d+/").unwrap();
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
        let filename = intern_table
            .get(&self.filename)
            .map_or("(unknown)", |s| s.as_str());
        write!(
            f,
            "{}:{} in {}",
            encode_text(simplify_filename(filename)),
            self.line,
            encode_text(&self.name)
        )
    }
}

pub type StackSummary = Vec<FrameSummary>;

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum SymInt {
    Int(i64),
    Symbol(String),
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

#[derive(Debug, Deserialize, Serialize)]
pub struct CompilationMetricsMetadata {
    // Other information like frame_key, co_name, etc. are already in envelope
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
pub struct AOTAutogradBackwardCompilationMetricsContext<'e> {
    pub m: &'e AOTAutogradBackwardCompilationMetricsMetadata,
    pub css: &'static str,
    pub compile_id: String,
}

#[derive(Debug, Serialize)]
pub struct CompilationMetricsContext<'e> {
    pub m: &'e CompilationMetricsMetadata,
    pub css: &'static str,
    pub compile_id: String,
    pub stack_html: String,
    pub symbolic_shape_specializations: Vec<SymbolicShapeSpecializationContext>,
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
    Artifact(&'e ArtifactMetadata),
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
    pub inductor_output_code: Option<InductorOutputCodeMetadata>,
    pub compilation_metrics: Option<CompilationMetricsMetadata>,
    pub aot_autograd_backward_compilation_metrics:
        Option<AOTAutogradBackwardCompilationMetricsMetadata>,
    pub graph_dump: Option<GraphDumpMetadata>,
    pub link: Option<LinkMetadata>,
    pub symbolic_shape_specialization: Option<SymbolicShapeSpecializationMetadata>,
    pub artifact: Option<ArtifactMetadata>,
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
    pub directory: Vec<(String, Vec<(String, String, i32)>)>,
    pub stack_trie_html: String,
    pub unknown_stack_trie_html: String,
    pub has_unknown_stack_trie: bool,
    pub num_breaks: usize,
    pub custom_header_html: String,
}

#[derive(Debug, Serialize)]
pub struct SymbolicShapeSpecializationContext {
    pub symbol: String,
    pub sources: Vec<String>,
    pub value: String,
    pub user_stack_html: String,
    pub stack_html: String,
}
