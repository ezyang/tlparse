use anyhow::anyhow;
use fxhash::FxHashMap;
use md5::{Digest, Md5};
use std::ffi::{OsStr, OsString};

use regex::Regex;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::path::PathBuf;
use tinytemplate::TinyTemplate;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::time::Instant;


use crate::types::*;
use crate::templates::*;
use crate::parsers::all_parsers;
mod parsers;
mod templates;
mod types;


pub struct ParseConfig {
    pub strict: bool,
}

pub fn parse_path(path: &PathBuf, config: ParseConfig) -> anyhow::Result<ParseOutput> {
    let strict = config.strict;
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

    // Store results in an output Vec<PathBuf, String>
    let mut output : Vec<(PathBuf, String)> = Vec::new();

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

    let all_parsers = all_parsers();

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

        let subdir = &compile_id_dir;

        let mut payload = String::new();
        if let Some(ref expect) = e.has_payload {
            let mut first = true;
            while let Some((_payload_lineno, payload_line)) =
                iter.next_if(|(_, l)| l.starts_with('\t'))
            {
                // Careful! Distinguish between missing EOL and not
                if !first {
                    payload.push('\n');
                }
                first = false;
                payload.push_str(&payload_line[1..]);
            }
            let mut hasher = Md5::new();
            hasher.update(&payload);
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

        stats.ok += 1;

        // lol this clone, probably shouldn't use entry
        // TODO: output should be able to generate this without explicitly creating
        let compile_directory = directory.entry(e.compile_id.clone()).or_default();

        for parser in &all_parsers {
            if let Some(md) = parser.get_metadata(&e) {
                let results = parser.parse(lineno, md, e.rank, &e.compile_id, &payload)?;
                for (filename, out) in results {
                    output.push((filename.clone(), out));
                    compile_directory.push(filename);
                }
            }
        }


        // TODO: implement these as StructuredLogParseres
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

        if e.dynamo_guards.is_some() {
            let filename = "dynamo_guards.html";
            let f = subdir.join(filename);
            match serde_json::from_str::<Vec<DynamoGuard>>(payload.as_str()) {
                Ok(guards) => {
                    let guards_context = DynamoGuardsContext { guards };
                    output.push((f, tt.render("dynamo_guards.html", &guards_context)?));
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
            output.push((f, payload.clone()));
            compile_directory.push(compile_id_dir.join(filename));
        }

        if let Some(metadata) = e.optimize_ddp_split_child {
            let filename = format!("optimize_ddp_split_child_{}.txt", metadata.name);
            let f = subdir.join(&filename);
            output.push((f, payload.clone()));
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
    output.push((PathBuf::from("index.html"), tt.render("index.html", &index_context)?));

    // other_rank is included here because you should only have logs from one rank when
    // configured properly
    if strict
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

    Ok(output)
}
