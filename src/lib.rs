use anyhow::anyhow;
use fxhash::FxHashMap;
use md5::{Digest, Md5};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use regex::Regex;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::PathBuf;
use std::time::Instant;
use tinytemplate::TinyTemplate;

use crate::parsers::default_parsers;
use crate::templates::*;
use crate::types::*;
mod parsers;
mod templates;
mod types;

pub struct ParseConfig {
    pub strict: bool,
    pub strict_compile_id: bool,
    pub custom_parsers: Vec<Box<dyn crate::parsers::StructuredLogParser>>,
}

fn maybe_remove_suffix(frames: &mut Vec<FrameSummary>) {
    let target_frames = [
        ("torch/_dynamo/convert_frame.py", "catch_errors"),
        ("torch/_dynamo/convert_frame.py", "_convert_frame"),
        ("torch/_dynamo/convert_frame.py", "_convert_frame_assert"),
    ];

    let len = frames.len();
    if len >= target_frames.len() {
        let suffix = &frames[len - target_frames.len()..];
        if suffix
            .iter()
            .zip(target_frames.iter())
            .all(|(frame, target)| {
                simplify_filename(unintern_str(frame.filename).as_ref()) == target.0
                    && frame.name == target.1
            })
        {
            frames.truncate(len - 3);
        }
    }
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
    let mut unknown_stack_trie = StackTrieNode::default();

    let mut stats = Stats::default();
    let _mod_count: FxHashMap<String, i32> = FxHashMap::default();

    let mut bytes_read: u64 = 0;

    // Some stuff for profiling
    let mut fastest_time = std::time::Duration::MAX;
    let mut slowest_time = std::time::Duration::ZERO;

    let mut expected_rank: Option<Option<u32>> = None;

    let mut directory: FxIndexMap<Option<CompileId>, Vec<(PathBuf, i32)>> = FxIndexMap::default();

    let mut metrics_index: CompilationMetricsIndex = FxIndexMap::default();

    // Store results in an output Vec<PathBuf, String>
    let mut output: Vec<(PathBuf, String)> = Vec::new();

    let mut tt: TinyTemplate = TinyTemplate::new();
    tt.add_formatter("format_unescaped", tinytemplate::format_unescaped);
    tt.add_template("index.html", TEMPLATE_INDEX)?;
    tt.add_template("failures_and_restarts.html", TEMPLATE_FAILURES_AND_RESTARTS)?;
    tt.add_template("dynamo_guards.html", TEMPLATE_DYNAMO_GUARDS)?;
    tt.add_template("compilation_metrics.html", TEMPLATE_COMPILATION_METRICS)?;
    tt.add_template(
        "aot_autograd_backward_compilation_metrics.html",
        TEMPLATE_AOT_AUTOGRAD_BACKWARD_COMPILATION_METRICS,
    )?;

    let mut output_count = 0;

    let mut breaks = RestartsAndFailuresContext {
        css: TEMPLATE_FAILURES_CSS,
        failures: Vec::new(),
    };

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

    let mut all_parsers = default_parsers(&tt);
    all_parsers.extend(config.custom_parsers);

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
                let results = parser.parse(lineno, md, e.rank, &e.compile_id, &payload);
                match results {
                    Ok(results) => {
                        for (filename, out) in results {
                            output.push((filename.clone(), out));
                            compile_directory.push((filename, output_count));
                            output_count += 1;
                        }
                    }
                    Err(err) => match parser.name() {
                        "dynamo_guards" => {
                            eprintln!("Failed to parse guards json: {}", err);
                            stats.fail_dynamo_guards_json += 1;
                        }
                        name => {
                            eprintln!("Parser {name} failed: {err}");
                            stats.fail_parser += 1;
                        }
                    },
                }
            }
        }

        if let Some(stack) = e.stack {
            unknown_stack_trie.insert(stack, None);
        }

        if let Some(m) = e.compilation_metrics {
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

            let id = e.compile_id.clone().map_or("(unknown) ".to_string(), |c| {
                format!(
                    "<a href='{}/compilation_metrics.html'>{cid}</a> ",
                    compile_id_dir.display(),
                    cid = c
                )
            });
            if let Some(rr) = m.restart_reasons.as_ref() {
                for restart in rr {
                    breaks.failures.push((
                        id.clone(),
                        format!("{}", FailureReason::Restart(restart.clone())),
                    ));
                }
            }
            if let Some(f) = m.fail_type.as_ref() {
                let reason = m
                    .fail_reason
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("Fail reason not found"))?;
                let user_frame_filename = m
                    .fail_user_frame_filename
                    .clone()
                    .unwrap_or(String::from("N/A"));
                let user_frame_lineno = m.fail_user_frame_lineno.unwrap_or(0);
                let failure_reason = FailureReason::Failure((
                    f.clone(),
                    reason.clone(),
                    user_frame_filename.clone(),
                    user_frame_lineno.clone(),
                ));
                breaks
                    .failures
                    .push((id.clone(), format!("{failure_reason}")));
            }
            let mut cid = e.compile_id.clone();
            if let Some(c) = cid.as_mut() {
                c.attempt = 0;
            }
            metrics_index.entry(cid).or_default().push(m);
        }

        if let Some(m) = e.dynamo_start {
            if let Some(mut stack) = m.stack {
                maybe_remove_suffix(&mut stack);
                stack_trie.insert(stack, e.compile_id.clone());
            };
        };
    }
    output.push((
        PathBuf::from("failures_and_restarts.html"),
        tt.render("failures_and_restarts.html", &breaks)?,
    ));
    pb.finish_with_message("done");
    spinner.finish();

    eprintln!("{:?}", stats);

    let has_unknown_compile_id = directory.contains_key(&None);

    let index_context = IndexContext {
        css: CSS,
        javascript: JAVASCRIPT,
        directory: directory
            .drain(..)
            .map(|(x, y)| (x.map_or("(unknown)".to_string(), |e| e.to_string()), y))
            .collect(),
        stack_trie_html: stack_trie.fmt(&metrics_index).unwrap(),
        unknown_stack_trie_html: unknown_stack_trie.fmt(&metrics_index).unwrap(),
        has_unknown_stack_trie: !unknown_stack_trie.is_empty(),
        num_breaks: breaks.failures.len(),
    };
    output.push((
        PathBuf::from("index.html"),
        tt.render("index.html", &index_context)?,
    ));

    // other_rank is included here because you should only have logs from one rank when
    // configured properly
    if strict
        && (stats.fail_glog
            + stats.fail_json
            + stats.fail_payload_md5
            + stats.other_rank
            + stats.fail_dynamo_guards_json
            + stats.fail_parser
            > 0)
    {
        // Report something went wrong
        return Err(anyhow!("Something went wrong"));
    }

    if config.strict_compile_id && has_unknown_compile_id {
        return Err(anyhow!("Some log entries did not have compile id"));
    }

    Ok(output)
}
