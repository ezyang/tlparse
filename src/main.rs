use clap::Parser;
use std::fmt::{self, Formatter, Display};
use core::hash::BuildHasherDefault;
use fxhash::{FxHashMap, FxHasher};
use indexmap::IndexMap;
use regex::Regex;
use std::fs::File;
use std::fs;
use std::io::{self, BufRead};
use std::path::PathBuf;
use tinytemplate::TinyTemplate;
use opener;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::time::Instant;

pub type FxIndexMap<K, V> = IndexMap<K, V, BuildHasherDefault<FxHasher>>;

static CSS: &'static str = r#"
body { font-family: monospace; }
"#;

static TEMPLATE_INDEX: &'static str = r#"
<html>
<style>
{css}
</style>
<body>
{stack_trie_html | format_unescaped}
</body>
</html>
"#;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    path: PathBuf,
    #[arg(short)]
    out: Option<PathBuf>,
    #[arg(long)]
    overwrite: bool,
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

    fn to_html(&self) -> String {
        let mut f = String::new();
        self.build_html(&mut f, 0);
        return f;
    }

    fn build_html(&self, f: &mut String, indent: usize) -> fmt::Result {
        for (frame, node) in self.children.iter() {
            let star = node.terminal.join("");
            if self.children.len() > 1 {
                // If the node has multiple children, increase the indent and print a hyphen
                write!(
                    f,
                    "{:indent$}- {star}",
                    "",
                    indent = indent,
                    star = star
                );
                tinytemplate::escape(frame, f);
                node.fmt_html(f, indent + 2);
            } else {
                // If the node has only one child, don't increase the indent and don't print a hyphen
                write!(
                    f,
                    "{:indent$}  {star}",
                    "",
                    indent = indent,
                    star = star
                );
                tinytemplate::escape(frame, f);
                node.fmt_html(f, indent);
            }
        }
    }
}

#[derive(Default, Debug)]
struct Stats {
    ok: u64,
    other_rank: u64,
    fail_glog: u64,
    fail_json: u64,
    compile_stack: u64,
}

#[derive(Debug, Hash, Eq, PartialEq, Deserialize)]
struct FrameSummary {
    filename: String,
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
        write!(
            f,
            "{}:{} in {}",
            simplify_filename(&self.filename),
            self.line,
            self.name
        )
    }
}

type StackSummary = Vec<FrameSummary>;

#[derive(Debug, Deserialize)]
struct Envelope {
    rank: Option<u32>,
    frame_id: Option<u32>,
    frame_compile_id: Option<u32>,
    attempt: Option<u32>,
    // externally tagged union, one field per log type we recognize
    compile_stack: Option<StackSummary>,
}

#[derive(Debug, Serialize)]
struct IndexContext {
    css: &'static str,
    stack_trie_html: String,
}

fn main() {
    let cli = Cli::parse();
    let path = cli.path;
    let out_path = cli.out.unwrap_or(PathBuf::from("tl_out"));

    if out_path.exists() {
        if !cli.overwrite {
            panic!("{} already exists, pass --overwrite to overwrite", out_path.display());
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

    for line in reader.lines() {
        let line = line.unwrap();
        bytes_read += line.len() as u64;
        pb.set_position(bytes_read);
        spinner.set_message(format!("{:?}", stats));
        //spinner.set_message(format!("{:?} {:?}", slowest_time, fastest_time));
        let start = Instant::now();

        let Some(caps) = re_glog.captures(&line) else {
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

        let envelope = match serde_json::from_str::<Envelope>(payload) {
            Ok(r) => r,
            Err(err) => {
                multi.suspend(|| { eprintln!("{}\n{:?}", payload, err); });
                stats.fail_json += 1;
                continue;
            }
        };

        match expected_rank {
            Some(rank) => {
                if rank != envelope.rank {
                    stats.other_rank += 1;
                    continue;
                }
            }
            None => {
                multi.suspend(|| { eprintln!("Detected rank: {:?}", envelope.rank); });
                expected_rank = Some(envelope.rank);
            }
        };

        stats.ok += 1;
        if let Some(stack) = envelope.compile_stack {
            stats.compile_stack += 1;
            stack_trie.insert(stack, "*".to_string()); // TODO: compile id
        };
    }
    pb.finish_with_message("done");
    spinner.finish();

    eprintln!("{:?}", stats);
    stack_trie.print(0);

    let mut tt = TinyTemplate::new();
    tt.add_template("index.html", TEMPLATE_INDEX);
    let index_context = IndexContext {
        css: CSS,
        stack_trie_html: stack_trie.to_html(),
    };
    fs::write(out_path.join("index.html"), tt.render("index.html", &index_context).unwrap()).unwrap();

    opener::open(out_path.join("index.html")).unwrap();

}
