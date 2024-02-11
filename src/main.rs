use clap::{Parser, Subcommand};
use regex::Regex;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

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
    writers: HashMap<u32, io::BufWriter<File>>,
}

impl RankDemuxer {
    fn new(out_dir: PathBuf) -> Self {
        RankDemuxer {
            out_dir: out_dir,
            writers: HashMap::new(),
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

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Summary { path } => summary(path),
    }
}

fn summary(path: &PathBuf) {
    print!("hello {}\n", path.display());
    let file = File::open(path).unwrap();
    let reader = io::BufReader::new(file);

    let re = Regex::new(concat!(
        r"(\[trainer\d+\]:)?(\[rank(?<rank>\d+)\]:)?",
        r"\[(?<year>\d{4})-(?<month>\d{2})-(?<day>\d{2}) ",
        r"(?<hour>\d{2}):(?<minute>\d{2}):(?<second>\d{2}),(?<millisecond>\d{3})\] ",
        r"(?<compile_id>(\[(?<frame_id>\d+)/(?<frame_compile_id>\d)+(_(?<restart>\d+))?\] )?)",
        r"(?<module>[^:]+): ",
        r"\[(?<level>DEBUG|INFO|WARNING|ERROR)\]",
        r" ?(?<message>.+)"
    ))
    .unwrap();

    let re2 = Regex::new(r"\[rank\d+\]:.+torch").unwrap();
    let mut ok = 0;
    let mut other_rank = 0;
    let mut no_rank = 0;
    let mut fail = 0;
    let mut skip = 0;
    let mut mod_count: HashMap<String, i32> = HashMap::new();
    let mut rank_demuxer = RankDemuxer::new(PathBuf::from("out")); // TODO: flag

    reader.lines().for_each(|line| {
        let line = line.unwrap();
        match re.captures(&line) {
            Some(caps) => {
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
                        ok += 1;
                        /*
                        if module == "torch._dynamo.guards.__guards" {
                            println!("{}", caps.name("message").unwrap().as_str())
                        }
                        */
                        let val = mod_count.entry(module.to_string()).or_insert(0);
                        *val += 1;
                    }
                    Some(_) => {
                        other_rank += 1;
                    }
                    None => {
                        no_rank += 1;
                    }
                }
            }
            None => {
                if re2.is_match(&line) {
                    println!("{}", line);
                    fail += 1;
                } else {
                    skip += 1;
                }
            }
        }
    });

    println!(
        "ok = {}, other_rank = {}, no_rank = {}, fail = {}, skip = {}",
        ok, other_rank, no_rank, fail, skip
    );
    for (key, value) in mod_count {
        println!("{}: {}", key, value);
    }
}
