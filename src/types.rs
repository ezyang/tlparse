use core::hash::BuildHasherDefault;
use fxhash::{FxHashMap, FxHasher};
use html_escape::encode_text;
use indexmap::IndexMap;

use std::fmt::{self, Display,Formatter};
use std::path::PathBuf;

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

pub type FxIndexMap<K, V> = IndexMap<K, V, BuildHasherDefault<FxHasher>>;

pub static INTERN_TABLE: Lazy<Mutex<FxHashMap<u32, String>>> =
    Lazy::new(|| Mutex::new(FxHashMap::default()));


#[derive(Default)]
pub struct StackTrieNode {
    terminal: Vec<String>,
    // Ordered map so that when we print we roughly print in chronological order
    children: FxIndexMap<FrameSummary, StackTrieNode>,
}

impl StackTrieNode {
    pub fn insert(&mut self, mut stack: StackSummary, compile_id: String) {
        let mut cur = self;
        for frame in stack.drain(..) {
            cur = cur.children.entry(frame).or_default();
        }
        cur.terminal.push(compile_id);
    }

    pub fn fmt_inner(&self, f: &mut Formatter, indent: usize) -> fmt::Result {
        for (frame, node) in self.children.iter() {
            let star = node.terminal.join("");
            if self.children.len() > 1 {
                // If the node has multiple children, increase the indent and print a hyphen
                writeln!(
                    f,
                    "{:indent$}- {star}{}",
                    "",
                    frame,
                    indent = indent,
                    star = star
                )?;
                node.fmt_inner(f, indent + 2)?;
            } else {
                // If the node has only one child, don't increase the indent and don't print a hyphen
                writeln!(
                    f,
                    "{:indent$}  {star}{}",
                    "",
                    frame,
                    indent = indent,
                    star = star
                )?;
                node.fmt_inner(f, indent)?;
            }
        }
        Ok(())
    }
}

impl Display for StackTrieNode {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "<pre>")?;
        self.fmt_inner(f, 0)?;
        write!(f, "</pre>")?;
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
}

#[derive(Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct FrameSummary {
    filename: u32,
    line: i32,
    name: String,
}

fn simplify_filename<'a>(filename: &'a str) -> &'a str {
    let parts: Vec<&'a str> = filename.split("#link-tree/").collect();
    if parts.len() > 1 {
        return parts[1];
    }
    // TODO: generalize this
    let parts: Vec<&'a str> = filename
        .split("1e322330-seed-nspid4026531836_cgpid26364902-ns-4026531840/")
        .collect();
    if parts.len() > 1 {
        return parts[1];
    }
    filename
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
pub struct OptimizeDdpSplitChildMetadata {
    pub name: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum SymInt {
    Int(i64),
    Symbol(String),
}

#[derive(Debug, Deserialize)]
pub struct EmptyMetadata {}

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
pub struct Envelope {
    pub rank: Option<u32>,
    #[serde(flatten)]
    pub compile_id: Option<CompileId>,
    #[serde(default)]
    pub has_payload: Option<String>,
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
    pub directory: Vec<(String, Vec<PathBuf>)>,
    pub stack_trie_html: String,
}
