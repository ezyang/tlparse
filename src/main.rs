use clap::{Parser, Subcommand};
use regex::Regex;
use core::hash::BuildHasherDefault;
use fxhash::{FxHashMap, FxHasher};
use indexmap::IndexMap;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::mem;
use std::fmt;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

pub type FxIndexMap<K, V> = IndexMap<K, V, BuildHasherDefault<FxHasher>>;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Prints some basic summary information about the log file
    Summary { path: PathBuf },
}

struct RankDemuxer {
    out_dir: PathBuf,
    writers: FxHashMap<u32, io::BufWriter<File>>,
}

impl RankDemuxer {
    fn new(out_dir: PathBuf) -> Self {
        RankDemuxer {
            out_dir: out_dir,
            writers: FxHashMap::default(),
        }
    }

    fn get(&mut self, rank: u32) -> &mut io::BufWriter<File> {
        if !self.writers.contains_key(&rank) {
            let file = File::create(self.out_dir.join(format!("rank{}.log", rank))).unwrap();
            let writer = io::BufWriter::new(file);
            self.writers.insert(rank, writer);
        }
        self.writers.get_mut(&rank).unwrap()
    }

    fn write(&mut self, rank: u32, log: &str) {
        writeln!(self.get(rank), "{}", log).unwrap();
    }
}

#[derive(Default)]
struct StackTrieNode {
    terminal: bool,
    // Ordered map so that when we print we roughly print in chronological order
    children: FxIndexMap<FrameSummary, StackTrieNode>,
}

impl StackTrieNode {
    fn insert(&mut self, mut stack: StackSummary) {
        let mut cur = self;
        for frame in stack.drain(..) {
            if frame.filename.contains("torch/_dynamo/eval_frame.py") && frame.name == "catch_errors" {
                break;
            }
            cur = cur.children.entry(frame).or_insert_with(|| StackTrieNode::default());
        }
        cur.terminal = true;
    }

    fn print(&self, indent: usize) {
        for (frame, node) in self.children.iter() {
            let mut star = "";
            if node.terminal {
                star = "⚡ ";
            }
            if self.children.len() > 1 {
                // If the node has multiple children, increase the indent and print a hyphen
                println!("{:indent$}- {star}{}", "", frame, indent = indent, star = star);
                node.print(indent + 2);
            } else {
                // If the node has only one child, don't increase the indent and don't print a hyphen
                println!("{:indent$}  {star}{}", "", frame, indent = indent, star = star);
                node.print(indent);
            }
        }
    }
}

#[derive(Default, Debug)]
struct Stats {
    ok: u64,
    other_rank: u64,
    no_rank: u64,
    fail: u64,
    skip: u64,
    stack_ok: u64,
    stack_truncated: u64,
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Summary { path } => summary(path),
    }
}

#[derive(Debug, Hash, Eq, PartialEq)]
struct FrameSummary {
    filename: String,
    lineno: i32,
    name: String,
}

fn simplify_filename<'a>(filename: &'a str) -> &'a str {
    let parts: Vec<&'a str> = filename.split("#link-tree/").collect();
    if parts.len() > 1 {
        return parts[1];
    }
    return filename;
}

impl fmt::Display for FrameSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{} in {}", simplify_filename(&self.filename), self.lineno, self.name)
    }
}

type StackSummary = Vec<FrameSummary>;

enum State {
    Scan,
    ExpectStackHeader,
    ExpectStackFile,
    ExpectStackCode,
}

fn summary(path: &PathBuf) {
    print!("hello {}\n", path.display());
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

    let mut st = State::Scan;

    let re_envelope = Regex::new(concat!(
        r"^(\[trainer\d+\]:)?(\[rank(?<rank>\d+)\]:)?",
        r"\[(?<year>\d{4})-(?<month>\d{2})-(?<day>\d{2}) ",
        r"(?<hour>\d{2}):(?<minute>\d{2}):(?<second>\d{2}),(?<millisecond>\d{3})\] ",
        r"(?<compile_id>(\[(?<frame_id>\d+)/(?<frame_compile_id>\d)+(_(?<restart>\d+))?\] )?)",
        r"(?<module>[^:]+): ",
        r"\[(?<level>DEBUG|INFO|WARNING|ERROR)\]",
        r" ?(?<message>.+)$"
    ))
    .unwrap();
    /*

    let re_envelope = Regex::new(concat!(
        r"(\[trainer\d+\]:)?(\[rank(?<rank>\d+)\]:)?",
        r"\[\d{4}-\d{2}-\d{2} ",
        r"\d{2}:\d{2}:\d{2},\d{3}\] ",
        r"(?<compile_id>(\[(?<frame_id>\d+)/(?<frame_compile_id>\d)+(_(?<restart>\d+))?\] )?)",
        r"(?<module>[^:]+): ",
        r"\[(?:DEBUG|INFO|WARNING|ERROR)\]",
        r" ?(?<message>.+)"
    ))
    .unwrap();
    */

    let re_fuzzy_envelope = Regex::new(r"\[rank\d+\]:.+torch").unwrap();

    let re_dynamo_start_tracing = Regex::new("Step 1: torchdynamo start tracing.+").unwrap();
    let re_stack_header = Regex::new(r"Stack.+:").unwrap();
    let re_stack_file = Regex::new(r#"  File "(?<file>[^"]+)", line (?<line>\d+), in (?<function>.+)"#).unwrap();
    let re_stack_code = Regex::new(r"    .+").unwrap();

    let mut stack: StackSummary = Vec::new();
    let mut stack_trie = StackTrieNode::default();

    let mut stats = Stats::default();
    let mut mod_count: FxHashMap<String, i32> = FxHashMap::default();
    let mut rank_demuxer = RankDemuxer::new(PathBuf::from("out")); // TODO: flag

    let mut bytes_read: u64 = 0;

    reader.lines().for_each(|line| {
        let line = line.unwrap();
        bytes_read += line.len() as u64;
        pb.set_position(bytes_read);
        spinner.set_message(format!("{:?}", stats));
        match re_envelope.captures(&line) {
            Some(caps) => {
                stats.ok += 1;
                return;
                let rank = caps
                    .name("rank")
                    .and_then(|v| v.as_str().parse::<u32>().ok());
                let compile_id = caps.name("compile_id").unwrap().as_str();
                let module = caps.name("module").unwrap().as_str();
                let level = caps.name("level").unwrap().as_str();
                let message = caps.name("message").unwrap().as_str();
                match rank {
                    Some(r) => {
                        rank_demuxer.write(
                            r,
                            &format!("{}{} [{}] {}", compile_id, module, level, message),
                        );
                    }
                    // These are safe to ignore, they're the top level launcher process logs
                    None => {}
                }
                match rank {
                    // TODO: make this configurable or automatically pick the rank with most log
                    // messages
                    Some(0) => {
                        stats.ok += 1;
                        /*
                        if module == "torch._dynamo.guards.__guards" {
                            println!("{}", caps.name("message").unwrap().as_str())
                        }
                        */
                        let val = mod_count.entry(module.to_string()).or_insert(0);
                        *val += 1;

                        // Run the state machine
                        let scan = |st: &mut State| {
                            if re_dynamo_start_tracing.is_match(message) {
                                *st = State::ExpectStackHeader;
                            }
                        };

                        let move_stack = |stack: &mut StackSummary, stack_trie: &mut StackTrieNode| {
                            if !stack.is_empty() {
                                let result_stack = mem::replace(stack, vec![]);
                                stack_trie.insert(result_stack);
                            }
                        };

                        match st {
                            State::Scan => { scan(&mut st); }
                            State::ExpectStackHeader => {
                                if re_stack_header.is_match(message) {
                                    st = State::ExpectStackFile;
                                    stack.clear();
                                } else {
                                    scan(&mut st);
                                }
                            }
                            State::ExpectStackFile => {
                                match re_stack_file.captures(message) {
                                    Some(caps) => {
                                        st = State::ExpectStackCode;
                                        let file = caps.name("file").unwrap().as_str();
                                        let line = caps.name("line").unwrap().as_str().parse::<i32>().ok().unwrap_or(-1);
                                        let function = caps.name("function").unwrap().as_str();
                                        stack.push(FrameSummary { filename: file.to_string(), lineno: line, name: function.to_string() });
                                    }
                                    None => {
                                        st = State::Scan;
                                        stats.stack_ok += 1;
                                        move_stack(&mut stack, &mut stack_trie);
                                        scan(&mut st);
                                    }
                                }
                            }
                            State::ExpectStackCode => {
                                if re_stack_code.is_match(message) {
                                    st = State::ExpectStackFile;
                                } else {
                                    st = State::Scan;
                                    stats.stack_truncated += 1;
                                    move_stack(&mut stack, &mut stack_trie);
                                    scan(&mut st);
                                }
                            }
                        }
                    }
                    Some(_) => {
                        stats.other_rank += 1;
                        //println!("{}", line);
                    }
                    None => {
                        stats.no_rank += 1;
                    }
                }
            }
            None => {
                if re_fuzzy_envelope.is_match(&line) {
                    // println!("{}", line);
                    stats.fail += 1;
                } else {
                    stats.skip += 1;
                }
            }
        }
    });
    pb.finish_with_message("done");

    println!("{:?}", stats);
    /*
    for (key, value) in mod_count {
        println!("{}: {}", key, value);
    }
    */

    stack_trie.print(0);
}
