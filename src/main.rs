use clap::{Parser, Subcommand};
use regex::Regex;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::PathBuf;
use std::collections::HashMap;

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
        r"(\[(?<frame_id>\d+)/(?<frame_compile_id>\d)+(_(?<restart>\d+))?\] )?",
        r"(?<module>[^:]+): ",
        r"\[(?<level>DEBUG|INFO|WARNING|ERROR)\]",
        r" ?(?<message>.+)"
    )).unwrap();

    let re2 = Regex::new(r"\[rank\d+\]:.+torch").unwrap();
    let mut ok = 0;
    let mut fail = 0;
    let mut skip = 0;
    let mut mod_count: HashMap<String, i32> = HashMap::new();

    reader.lines().for_each(|line| {
        let line = line.unwrap();
        match re.captures(&line) {
            Some(caps) => {
                ok += 1;
                let module = caps.name("module").unwrap().as_str();
                if (module == "torch._dynamo.guards.__guards") {
                    print!("{}\n", caps.name("message").unwrap().as_str())
                }
                let val = mod_count.entry(module.to_string()).or_insert(0);
                *val += 1;
            }
            None => {
                if re2.is_match(&line) {
                    print!("{}\n", line);
                    fail += 1;
                } else {
                    skip += 1;
                }
            }
        }
    });

    print!("ok = {}, fail = {}, skip = {}\n", ok, fail, skip);
    for (key, value) in mod_count {
        print!("{}: {}\n", key, value);
    }
}
