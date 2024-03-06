use anyhow::anyhow;
use base16ct;
use clap::Parser;
use core::hash::BuildHasherDefault;
use fxhash::{FxHashMap, FxHasher};
use html_escape::encode_text;
use indexmap::IndexMap;
use md5::{Digest, Md5};
use std::ffi::{OsStr, OsString};

use regex::Regex;
use std::fmt::{self, Display, Formatter};
use std::fs;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::path::PathBuf;
use tinytemplate::TinyTemplate;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use std::time::Instant;

pub type FxIndexMap<K, V> = IndexMap<K, V, BuildHasherDefault<FxHasher>>;

static INTERN_TABLE: Lazy<Mutex<FxHashMap<u32, String>>> =
    Lazy::new(|| Mutex::new(FxHashMap::default()));

static CSS: &str = r#"
"#;

static TEMPLATE_DYNAMO_GUARDS: &str = r#"
<html>
<body>
<h2>Guards</h2>
<ul>
{{ for guard in guards }}
    <li><code>{guard.code}</code></li>
{{ endfor }}
</ul>
</body>
</html>
"#;

static TEMPLATE_INDEX: &str = r#"
<html>
<style>
{css}
</style>
<body>
<div>
<h2>Stack trie</h2>
<p>
The <strong>stack trie</strong> is a way of getting a quick orientation on where all the
compilations in a model take place, esp., if you are compiling a codebase you are unfamiliar with.
It is a tree of stack frames, for all stacks that triggered PT2 compilation.  If only a single
stack is in the tree, you will simply see a plain list of frames (most recent call last).  With
multiple stacks, at every point where two stacks diverge from having a common prefix, we increase
the indentation of the list and have a separate sub-list per sub-tree.
</p>
{stack_trie_html | format_unescaped}
</div>
<div>
<h2>IR dumps</h2>
<p>
The <strong>IR dumps</strong> collected dumped intermediate products from various points of the PT2
compilation process.  The products are organized by compile id, and then sorted in chronological
order.
</p>
<p>
A <strong>compile id</strong> uniquely identifies are particular compilation inside a PT2
program.  It is traditionally written as <code>[x/y]</code>, where the <strong>frame id</strong> x
identifies the particular Python frame which we are compiling, and <strong>frame compile
id</strong> y identifies how many times we've recompiled this same frame.  For example,
<code>[0/0]</code> refers to the very first frame compiled by PT2; <code>[0/1]</code> refers to the
first recompilation of this frame, while <code>[1/0]</code> refers to a different frame, within
distinct code cache, which we are compiling next (perhaps because of a graph break).  Although
Dynamo treats distinct frames as completely unrelated, a frame compilation could overlap with another
frame; for example, if you graph break in an inlined function, Dynamo will typically try to compile
the nested frame again on an inner frame.  You can identify the hierarchical relationship between
frames by looking at the stack trie above.
</p>
<p>
In some situations, the compile id will have an extra signifier <code>[x/y_z]</code>, where z is the
<strong>attempt</strong> for this particular (re)compilation.  Certain conditions will cause Dynamo to
restart analysis, when Dynamo discovers that it needs to undo a decision it previously made.  The most
common cause of recompilation is a graph break in an inlined function call, which forces to restart
and avoid inlining the function in the first place.
</p>
<p>
Here is a high level description of PT2's compilation phases, and the intermediate products each
phase generates:
</p>
<ol>
<li><em>Optional:</em> If compiled autograd is enabled, and we are processing a backward call, compiled autograd will trace the autograd graph from the autograd engine, and produce an FX graph <code>compiled_autograd_graph</code> that will be Dynamo traced.  Otherwise, Dynamo will directly trace user's bytecode.</li>
<li>Dynamo symbolically evaluates the Python bytecode of a program, producing <code>dynamo_output_graph</code></li>
<li><em>Optional:</em> If <code>optimize_ddp</code> is enabled, the DDPOptimizer will split the Dynamo output graph to improve pipelining communications.  Each split subgraph is <code>optimize_ddp_split_child_submod</code>, and the high level graph that plumbs the graphs together is <code>optimize_ddp_split_graph</code>.  If there are multiple splits, each subsequent build product will be produced multiple times, one for each split.</li>
<li>AOTAutograd traces the (possibly split) Dynamo output graph, producing a <code>aot_joint_graph</code> if backwards is enabled.  It then partitions the graph into <code>aot_forward_graph</code> and <code>aot_backward_graph</code>.  If training is not needed, there may only be an <code>aot_forward_graph</code>.</li>
<li>Inductor will apply some post grad FX passes, producing <code>inductor_post_grad_graph</code></li>
<li>Inductor will perform code generation, producing the final <code>inductor_output_code</code> which will be executed at runtime.  This output is a valid Python program and can be directly run.</li>
</ol>
<p>
Build products below:
</p>
<ul>
{{ for compile_directory in directory }}
    <li><a id="{compile_directory.0}">{compile_directory.0}</a>
    <ul>
        {{ for path in compile_directory.1 }}
            <li><a href="{path}">{path}</a></li>
        {{ endfor }}
    </ul>
    </li>
{{ endfor }}
</ul>
</div>
</body>
</html>
"#;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    path: PathBuf,
    /// Output directory, defaults to `tl_out`
    #[arg(short, default_value = "tl_out")]
    out: PathBuf,
    /// Delete out directory if it already exists
    #[arg(long)]
    overwrite: bool,
    /// Return non-zero exit code if unrecognized log lines are found.  Mostly useful for unit
    /// testing.
    #[arg(long)]
    strict: bool,
    /// Don't open browser at the end
    #[arg(long)]
    no_browser: bool,
}

#[derive(Default)]
struct StackTrieNode {
    terminal: Vec<String>,
    // Ordered map so that when we print we roughly print in chronological order
    children: FxIndexMap<FrameSummary, StackTrieNode>,
}

impl StackTrieNode {
    fn insert(&mut self, mut stack: StackSummary, compile_id: String) {
        let mut cur = self;
        for frame in stack.drain(..) {
            cur = cur.children.entry(frame).or_default();
        }
        cur.terminal.push(compile_id);
    }

    fn fmt_inner(&self, f: &mut Formatter, indent: usize) -> fmt::Result {
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
struct CompileId {
    frame_id: u32,
    frame_compile_id: u32,
    attempt: u32,
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
struct Stats {
    ok: u64,
    other_rank: u64,
    fail_glog: u64,
    fail_json: u64,
    fail_payload_md5: u64,
    fail_dynamo_guards_json: u64,
}

#[derive(Debug, Hash, Eq, PartialEq, Deserialize, Serialize)]
struct FrameSummary {
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

type StackSummary = Vec<FrameSummary>;

#[derive(Debug, Deserialize)]
struct OptimizeDdpSplitChildMetadata {
    name: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum SymInt {
    Int(i64),
    Symbol(String),
}

#[derive(Debug, Deserialize)]
struct EmptyMetadata {}

#[derive(Debug, Deserialize)]
struct DynamoOutputGraphMetadata {
    _sizes: Option<FxHashMap<String, Vec<SymInt>>>,
}

#[derive(Debug, Deserialize)]
struct DynamoStartMetadata {
    stack: Option<StackSummary>,
}

#[derive(Debug, Deserialize)]
struct InductorOutputCodeMetadata {
    filename: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct Envelope {
    rank: Option<u32>,
    #[serde(flatten)]
    compile_id: Option<CompileId>,
    #[serde(default)]
    has_payload: Option<String>,
    // externally tagged union, one field per log type we recognize
    dynamo_start: Option<DynamoStartMetadata>,
    str: Option<(String, u32)>,
    dynamo_output_graph: Option<DynamoOutputGraphMetadata>,
    optimize_ddp_split_graph: Option<EmptyMetadata>,
    optimize_ddp_split_child: Option<OptimizeDdpSplitChildMetadata>,
    compiled_autograd_graph: Option<EmptyMetadata>,
    dynamo_guards: Option<EmptyMetadata>,
    aot_forward_graph: Option<EmptyMetadata>,
    aot_backward_graph: Option<EmptyMetadata>,
    aot_joint_graph: Option<EmptyMetadata>,
    inductor_post_grad_graph: Option<EmptyMetadata>,
    inductor_output_code: Option<InductorOutputCodeMetadata>,
}

#[derive(Debug, Deserialize, Serialize)]
struct DynamoGuard {
    code: String,
    stack: Option<StackSummary>,
    user_stack: Option<StackSummary>,
}

#[derive(Debug, Serialize)]
struct DynamoGuardsContext {
    guards: Vec<DynamoGuard>,
}

#[derive(Debug, Serialize)]
struct IndexContext {
    css: &'static str,
    directory: Vec<(String, Vec<PathBuf>)>,
    stack_trie_html: String,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let path = cli.path;
    let out_path = cli.out;

    if out_path.exists() {
        if !cli.overwrite {
            panic!(
                "{} already exists, pass --overwrite to overwrite",
                out_path.display()
            );
        }
        fs::remove_dir_all(&out_path)?;
    }
    fs::create_dir(&out_path)?;

    let file = File::open(path)?;
    let metadata = file.metadata()?;
    let file_size = metadata.len();
    let multi = MultiProgress::new();
    let pb = multi.add(ProgressBar::new(file_size));
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} [{bytes_per_sec}] ({eta})")?
        .progress_chars("#>-"));
    let spinner = multi.add(ProgressBar::new_spinner());
    let reader = io::BufReader::new(file);

    let re_glog = Regex::new(concat!(
        r"(?<level>[VIWEC])(?<month>\d{2})(?<day>\d{2}) ",
        r"(?<hour>\d{2}):(?<minute>\d{2}):(?<second>\d{2}).(?<millisecond>\d{6}) ",
        r"(?<thread>\d+)",
        r"(?<pathname>[^:]+):(?<line>\d+)\] ",
        r"(?<payload>.)"
    ))?;

    let mut stack_trie = StackTrieNode::default();

    let mut stats = Stats::default();
    let _mod_count: FxHashMap<String, i32> = FxHashMap::default();

    let mut bytes_read: u64 = 0;

    // Some stuff for profiling
    let mut fastest_time = std::time::Duration::MAX;
    let mut slowest_time = std::time::Duration::ZERO;

    let mut expected_rank: Option<Option<u32>> = None;

    let mut directory: FxHashMap<Option<CompileId>, Vec<PathBuf>> = FxHashMap::default();

    let mut tt = TinyTemplate::new();
    tt.add_formatter("format_unescaped", tinytemplate::format_unescaped);
    tt.add_template("index.html", TEMPLATE_INDEX)?;
    tt.add_template("dynamo_guards.html", TEMPLATE_DYNAMO_GUARDS)?;

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

    while let Some((lineno, line)) = iter.next() {
        bytes_read += line.len() as u64;
        pb.set_position(bytes_read);
        spinner.set_message(format!("{:?}", stats));
        //spinner.set_message(format!("{:?} {:?}", slowest_time, fastest_time));
        let start = Instant::now();

        let Some(caps) = re_glog.captures(&line) else {
            eprintln!("Failed to parse glog prefix on line {}", lineno);
            stats.fail_glog += 1;
            continue;
        };

        let end = start.elapsed();
        if end < fastest_time {
            fastest_time = end;
        }
        if end > slowest_time {
            slowest_time = end;
            //println!("{}", line);
        }
        let payload = &line[caps.name("payload").unwrap().start()..];

        let e = match serde_json::from_str::<Envelope>(payload) {
            Ok(r) => r,
            Err(err) => {
                multi.suspend(|| {
                    eprintln!("Failed to parse metadata JSON: {}\n{:?}", payload, err);
                });
                stats.fail_json += 1;
                continue;
            }
        };

        if let Some((s, i)) = e.str {
            let mut intern_table = INTERN_TABLE.lock().unwrap();
            intern_table.insert(i, s);
            continue;
        };

        match expected_rank {
            Some(rank) => {
                if rank != e.rank {
                    stats.other_rank += 1;
                    continue;
                }
            }
            None => {
                multi.suspend(|| {
                    eprintln!("Detected rank: {:?}", e.rank);
                });
                expected_rank = Some(e.rank);
            }
        };

        let compile_id_dir: PathBuf = e
            .compile_id
            .as_ref()
            .map_or(
                format!("unknown_{lineno}"),
                |CompileId {
                     frame_id,
                     frame_compile_id,
                     attempt,
                 }| { format!("{frame_id}_{frame_compile_id}_{attempt}") },
            )
            .into();

        let subdir = out_path.join(&compile_id_dir);
        fs::create_dir_all(&subdir)?;

        let mut payload = String::new();
        if let Some(expect) = e.has_payload {
            let mut first = true;
            while let Some((_payload_lineno, payload_line)) =
                iter.next_if(|(_, l)| l.starts_with('\t'))
            {
                // Careful! Distinguish between missing EOL and not
                if !first {
                    payload.push_str("\n");
                }
                first = false;
                payload.push_str(&payload_line[1..]);
            }
            let mut hasher = Md5::new();
            hasher.update(&payload);
            let hash = hasher.finalize();
            let mut expect_buf = [0u8; 16];
            if let Ok(_) = base16ct::lower::decode(expect, &mut expect_buf) {
                if expect_buf != &hash[..] {
                    // TODO: error log
                    stats.fail_payload_md5 += 1;
                }
            } else {
                stats.fail_payload_md5 += 1;
            }
        }

        stats.ok += 1;

        // lol this clone, probably shouldn't use entry
        let compile_directory = directory.entry(e.compile_id.clone()).or_default();

        if let Some(m) = e.dynamo_start {
            if let Some(stack) = m.stack {
                stack_trie.insert(
                    stack,
                    e.compile_id.map_or("(unknown) ".to_string(), |c| {
                        format!("<a href='#{cid}'>{cid}</a> ", cid = c)
                    }),
                );
            };
        };

        let mut write_dump =
            |filename: &str, sentinel: Option<EmptyMetadata>| -> anyhow::Result<()> {
                if sentinel.is_some() {
                    let f = subdir.join(filename);
                    fs::write(&f, &payload)?;
                    compile_directory.push(compile_id_dir.join(filename));
                }
                Ok(())
            };

        write_dump("optimize_ddp_split_graph.txt", e.optimize_ddp_split_graph)?;
        write_dump("compiled_autograd_graph.txt", e.compiled_autograd_graph)?;
        write_dump("aot_forward_graph.txt", e.aot_forward_graph)?;
        write_dump("aot_backward_graph.txt", e.aot_backward_graph)?;
        write_dump("aot_joint_graph.txt", e.aot_joint_graph)?;
        write_dump("inductor_post_grad_graph.txt", e.inductor_post_grad_graph)?;

        if e.dynamo_output_graph.is_some() {
            // TODO: dump sizes
            let filename = "dynamo_output_graph.txt";
            let f = subdir.join(&filename);
            fs::write(&f, &payload)?;
            compile_directory.push(compile_id_dir.join(filename));
        }

        if e.dynamo_guards.is_some() {
            let filename = "dynamo_guards.html";
            let f = subdir.join(&filename);
            match serde_json::from_str::<Vec<DynamoGuard>>(payload.as_str()) {
                Ok(guards) => {
                    let guards_context = DynamoGuardsContext { guards: guards };
                    fs::write(&f, tt.render("dynamo_guards.html", &guards_context)?)?;
                    compile_directory.push(compile_id_dir.join(filename));
                }
                Err(err) => {
                    eprintln!("Failed to parse guards json: {}", err);
                    stats.fail_dynamo_guards_json += 1;
                }
            }
        }

        if let Some(metadata) = e.inductor_output_code {
            let filename = metadata
                .filename
                .as_ref()
                .and_then(|p| Path::file_stem(p))
                .map_or_else(
                    || PathBuf::from("inductor_output_code.txt"),
                    |stem| {
                        let mut r = OsString::from("inductor_output_code_");
                        r.push(stem);
                        r.push(OsStr::new(".txt"));
                        r.into()
                    },
                );
            let f = subdir.join(&filename);
            fs::write(&f, &payload)?;
            compile_directory.push(compile_id_dir.join(filename));
        }

        if let Some(metadata) = e.optimize_ddp_split_child {
            let filename = format!("optimize_ddp_split_child_{}.txt", metadata.name);
            let f = subdir.join(&filename);
            fs::write(&f, &payload)?;
            compile_directory.push(compile_id_dir.join(filename));
        }
    }
    pb.finish_with_message("done");
    spinner.finish();

    eprintln!("{:?}", stats);

    let index_context = IndexContext {
        css: CSS,
        directory: directory
            .drain()
            .map(|(x, y)| (x.map_or("(unknown)".to_string(), |e| e.to_string()), y))
            .collect(),
        stack_trie_html: stack_trie.to_string(),
    };
    fs::write(
        out_path.join("index.html"),
        tt.render("index.html", &index_context)?,
    )?;

    if !cli.no_browser {
        opener::open(out_path.join("index.html"))?;
    }

    // other_rank is included here because you should only have logs from one rank when
    // configured properly
    if cli.strict
        && (stats.fail_glog
            + stats.fail_json
            + stats.fail_payload_md5
            + stats.other_rank
            + stats.fail_dynamo_guards_json
            > 0)
    {
        // Report something went wrong
        return Err(anyhow!("Something went wrong"));
    }

    Ok(())
}
