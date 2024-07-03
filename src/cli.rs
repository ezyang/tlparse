use clap::Parser;

use std::fs;
use std::path::PathBuf;

use tlparse::{parse_path, ParseConfig};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
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
    /// Return non-zero exit code if some log lines do not have associated compile id.  Used for
    /// unit testing
    #[arg(long)]
    strict_compile_id: bool,
    /// Don't open browser at the end
    #[arg(long)]
    no_browser: bool,
    /// Some custom HTML to append to the top of report
    #[arg(long, default_value = "")]
    custom_header_html: String,
    /// Be more chatty
    #[arg(short, long)]
    verbose: bool,
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

    let config = ParseConfig {
        strict: cli.strict,
        strict_compile_id: cli.strict_compile_id,
        custom_parsers: Vec::new(),
        custom_header_html: cli.custom_header_html,
        verbose: cli.verbose,
    };

    let output = parse_path(&path, config)?;

    for (filename, path) in output {
        let out_file = out_path.join(filename);
        if let Some(dir) = out_file.parent() {
            fs::create_dir_all(dir)?;
        }
        fs::write(out_file, path)?;
    }

    if !cli.no_browser {
        opener::open(out_path.join("index.html"))?;
    }
    Ok(())
}
