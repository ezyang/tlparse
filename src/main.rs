use base16ct;
use clap::Parser;
use core::hash::BuildHasherDefault;
use fxhash::{FxHashMap, FxHasher};
use html_escape::encode_text;
use indexmap::IndexMap;
use md5::{Digest, Md5};

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
use std::process::ExitCode;
use std::sync::Mutex;
use std::time::Instant;

pub type FxIndexMap<K, V> = IndexMap<K, V, BuildHasherDefault<FxHasher>>;

static INTERN_TABLE: Lazy<Mutex<FxHashMap<u32, String>>> =
    Lazy::new(|| Mutex::new(FxHashMap::default()));

static CSS: &str = r#"
"#;

static TEMPLATE_INDEX: &str = r#"
<html>
<style>
{css}
</style>
<body>
<div>
<h2>Stack trie</h2>
{stack_trie_html | format_unescaped}
</div>
<div>
<h2>IR dumps</h2>
<ul>
{{ for compile_directory in directory }}
    <li>{compile_directory.0}
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
    #[arg(short)]
    out: Option<PathBuf>,
    /// Delete out directory if it already exists
    #[arg(long)]
    overwrite: bool,
    /// Return non-zero exit code if unrecognized log lines are found.  Mostly useful for unit
    /// testing.
    #[arg(long)]
    strict: bool,
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
}

#[derive(Debug, Hash, Eq, PartialEq, Deserialize)]
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
    sizes: Option<FxHashMap<String, Vec<SymInt>>>,
}

#[derive(Debug, Deserialize)]
struct DynamoStartMetadata {
    stack: Option<StackSummary>,
}

#[derive(Debug, Deserialize)]
struct InductorOutputCodeMetadata {
    filename: Option<String>,
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

#[derive(Debug, Serialize)]
struct IndexContext {
    css: &'static str,
    directory: Vec<(String, Vec<PathBuf>)>,
    stack_trie_html: String,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let path = cli.path;
    let out_path = cli.out.unwrap_or(PathBuf::from("tl_out"));

    if out_path.exists() {
        if !cli.overwrite {
            panic!(
                "{} already exists, pass --overwrite to overwrite",
                out_path.display()
            );
        }
        fs::remove_dir_all(&out_path).unwrap();
    }
    fs::create_dir(&out_path).unwrap();

    let file = File::open(path).unwrap();
    let metadata = file.metadata().unwrap();
    let file_size = metadata.len();
    let multi = MultiProgress::new();
    let pb = multi.add(ProgressBar::new(file_size));
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} [{bytes_per_sec}] ({eta})").unwrap()
        .progress_chars("#>-"));
    let spinner = multi.add(ProgressBar::new_spinner());
    let reader = io::BufReader::new(file);

    let re_glog = Regex::new(concat!(
        r"(?<level>[VIWEC])(?<month>\d{2})(?<day>\d{2}) ",
        r"(?<hour>\d{2}):(?<minute>\d{2}):(?<second>\d{2}).(?<millisecond>\d{6}) ",
        r"(?<thread>\d+)",
        r"(?<pathname>[^:]+):(?<line>\d+)\] ",
        r"(?<payload>.)"
    ))
    .unwrap();

    let mut stack_trie = StackTrieNode::default();

    let mut stats = Stats::default();
    let _mod_count: FxHashMap<String, i32> = FxHashMap::default();

    let mut bytes_read: u64 = 0;

    // Some stuff for profiling
    let mut fastest_time = std::time::Duration::MAX;
    let mut slowest_time = std::time::Duration::ZERO;

    let mut expected_rank: Option<Option<u32>> = None;

    let mut directory: FxHashMap<Option<CompileId>, Vec<PathBuf>> = FxHashMap::default();

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
            eprintln!("fail_glog {}", lineno);
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
                    eprintln!("{}\n{:?}", payload, err);
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

        // TODO: borrow only here
        let compile_id_dir = e
            .compile_id
            .clone()
            .map_or(format!("unknown_{}", lineno), |e: CompileId| {
                format!("{}_{}_{}", e.frame_id, e.frame_compile_id, e.attempt)
            });

        let subdir = out_path.join(&compile_id_dir);
        fs::create_dir_all(&subdir).unwrap();

        let mut payload = String::new();
        if let Some(expect) = e.has_payload {
            let mut first = true;
            while let Some((payload_lineno, payload_line)) =
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
                    e.compile_id.map_or("* ".to_string(), |c| c.to_string()),
                );
            };
        };

        let mut write_dump = |filename: &str, sentinel: Option<EmptyMetadata>| {
            if let Some(_r) = sentinel {
                let f = subdir.join(filename);
                fs::write(&f, &payload).unwrap();
                compile_directory.push(Path::new(&compile_id_dir).join(filename));
            }
        };

        write_dump("optimize_ddp_split_graph.txt", e.optimize_ddp_split_graph);
        write_dump("compiled_autograd_graph.txt", e.compiled_autograd_graph);
        write_dump("aot_forward_graph.txt", e.aot_forward_graph);
        write_dump("aot_backward_graph.txt", e.aot_backward_graph);
        write_dump("aot_joint_graph.txt", e.aot_joint_graph);
        write_dump("inductor_post_grad_graph.txt", e.inductor_post_grad_graph);

        if let Some(_metadata) = e.dynamo_output_graph {
            // TODO: dump sizes
            let filename = "dynamo_output_graph.txt";
            let f = subdir.join(&filename);
            fs::write(&f, &payload).unwrap();
            compile_directory.push(Path::new(&compile_id_dir).join(filename));
        }

        if let Some(metadata) = e.inductor_output_code {
            let filename = match metadata.filename {
                Some(p) =>
                // Bah, where's pattern guards when you need 'em
                {
                    match Path::new(&p).file_stem() {
                        Some(stem) => {
                            format!("inductor_output_code_{}.txt", stem.to_str().unwrap())
                        }
                        None => "inductor_output_code.txt".to_string(),
                    }
                }
                None => "inductor_output_code.txt".to_string(),
            };
            let f = subdir.join(&filename);
            fs::write(&f, &payload).unwrap();
            compile_directory.push(Path::new(&compile_id_dir).join(filename));
        }

        if let Some(metadata) = e.optimize_ddp_split_child {
            let filename = format!("optimize_ddp_split_child_{}.txt", metadata.name);
            let f = subdir.join(&filename);
            fs::write(&f, &payload).unwrap();
            compile_directory.push(Path::new(&compile_id_dir).join(filename));
        }
    }
    pb.finish_with_message("done");
    spinner.finish();

    eprintln!("{:?}", stats);

    let mut tt = TinyTemplate::new();
    tt.add_formatter("format_unescaped", tinytemplate::format_unescaped);
    tt.add_template("index.html", TEMPLATE_INDEX).unwrap();
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
        tt.render("index.html", &index_context).unwrap(),
    )
    .unwrap();

    opener::open(out_path.join("index.html")).unwrap();

    // other_rank is included here because you should only have logs from one rank when
    // configured properly
    if cli.strict
        && (stats.fail_glog + stats.fail_json + stats.fail_payload_md5 + stats.other_rank > 0)
    {
        // Report something went wrong
        ExitCode::from(1)
    } else {
        ExitCode::from(0)
    }
}
