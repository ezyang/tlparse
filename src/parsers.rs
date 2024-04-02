use crate::types::*;
use std::path::PathBuf;
use std::ffi::{OsStr, OsString};
use std::path::Path;
use tinytemplate::TinyTemplate;

/**
 * StructuredLogParser
 * Parses a structured log and returns a vec of file outputs.
 * Implement this trait to add your own analyses.
 *
 * 'e is the lifetime of the envelope being parsed
 */
pub trait StructuredLogParser {
    // If this returns Some value, the parser will be run on that metadata.
    // Otherwise, it will be skipped.
    fn get_metadata<'e>(&self, e: &'e Envelope) -> Option<Metadata<'e>>;

    // Take a log input and the metadata you asked for, return a set of files to write
    fn parse<'e>(&self,
        lineno: usize, // Line number from log
        metadata: Metadata<'e>, // Metadata from get_metadata
        rank: Option<u32>, // Rank of the log
        compile_id: &Option<CompileId>, // Compile ID of the envelope
        payload: &str // Payload from the log (empty string when None)
    ) -> anyhow::Result<ParseOutput>;

    // Name of the parser, for error logging
    fn name(&self) -> &'static str;
}

// Takes a filename and a payload and writes that payload into a the file
fn simple_file_output(
    filename: &str,
    lineno: usize,
    compile_id: &Option<CompileId>,
    payload: &str
) -> anyhow::Result<ParseOutput> {
    let compile_id_dir: PathBuf = compile_id
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
    let subdir = PathBuf::from(compile_id_dir);
    let f = subdir.join(filename);
    Ok(Vec::from([(f, String::from(payload))]))
}

/**
 * Parser for simple output dumps where the metadata is a sentinel {}
 */
pub struct SentinelFileParser {
    filename: &'static str,
    get_sentinel: fn (&Envelope) -> Option<&EmptyMetadata>,
} impl SentinelFileParser {
    pub fn new(filename: &'static str, get_sentinel: fn (&Envelope) -> Option<&EmptyMetadata>) -> Self {
        Self { filename, get_sentinel }
    }
}
impl StructuredLogParser for SentinelFileParser {
    fn name(&self) -> &'static str {
        self.filename
    }
    fn get_metadata<'e>(&self, e: &'e Envelope) -> Option<Metadata<'e>> {
        (self.get_sentinel)(e).map(|m| Metadata::Empty(m))
    }
    fn parse<'e>(&self,
            lineno: usize,
            _metadata: Metadata<'e>,
            _rank: Option<u32>,
            compile_id: &Option<CompileId>,
            payload: &str
    ) -> anyhow::Result<ParseOutput> {
        simple_file_output(&format!("{}.txt",self.filename), lineno, compile_id, payload)
    }
}

// Same as SentinelFileParser, but can log the size of the graph
pub struct DynamoOutputGraphParser;
impl StructuredLogParser for DynamoOutputGraphParser {
    fn name(&self) -> &'static str {
        "dynamo_output_graph"
    }
    fn get_metadata<'e>(&self, e: &'e Envelope) -> Option<Metadata<'e>> {
        e.dynamo_output_graph.as_ref().map(|m| Metadata::DynamoOutputGraph(m))
    }
    fn parse<'e>(&self,
            lineno: usize,
            _metadata: Metadata<'e>, // TODO: log size of graph
            _rank: Option<u32>,
            compile_id: &Option<CompileId>,
            payload: &str
    ) -> anyhow::Result<ParseOutput> {
        simple_file_output("dynamo_output_graph.txt", lineno, compile_id, payload)
    }
}

pub struct DynamoGuardParser<'t> {
    tt: &'t TinyTemplate<'t>,
}
impl StructuredLogParser for DynamoGuardParser<'_> {
    fn name(&self) -> &'static str {
        "dynamo_guards"
    }
    fn get_metadata<'e>(&self, e: &'e Envelope) -> Option<Metadata<'e>> {
        e.dynamo_guards.as_ref().map(|m| Metadata::Empty(m))
    }
    fn parse<'e>(&self,
            lineno: usize,
            _metadata: Metadata<'e>,
            _rank: Option<u32>,
            compile_id: &Option<CompileId>,
            payload: &str
    ) -> anyhow::Result<ParseOutput> {
        let filename = format!("{}.html", self.name());
        let guards = serde_json::from_str::<Vec<DynamoGuard>>(payload)?;
        let guards_context = DynamoGuardsContext { guards };
        let output = self.tt.render(&filename, &guards_context)?;
        simple_file_output(&filename, lineno, compile_id, &output)
    }
}

pub struct InductorOutputCodeParser;
impl StructuredLogParser for InductorOutputCodeParser {
    fn name(&self) -> &'static str {
        "inductor_output_code"
    }
    fn get_metadata<'e>(&self, e: &'e Envelope) -> Option<Metadata<'e>> {
        e.inductor_output_code.as_ref().map(|m| Metadata::InductorOutputCode(m))
    }

    fn parse<'e>(&self,
        lineno: usize,
        metadata: Metadata<'e>,
        _rank: Option<u32>,
        compile_id: &Option<CompileId>,
        payload: &str
    ) -> anyhow::Result<ParseOutput> {
        if let Metadata::InductorOutputCode(metadata) = metadata {
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
            simple_file_output(&filename.to_string_lossy(), lineno, compile_id, payload)
        } else {
            Err(anyhow::anyhow!("Expected InductorOutputCode metadata"))
        }
    }
}

pub struct OptimizeDdpSplitChildParser;
impl StructuredLogParser for OptimizeDdpSplitChildParser {
    fn name(&self) -> &'static str {
        "optimize_ddp_split_child"
    }
    fn get_metadata<'e>(&self, e: &'e Envelope) -> Option<Metadata<'e>> {
        e.optimize_ddp_split_child.as_ref().map(|m| Metadata::OptimizeDdpSplitChild(m))
    }

    fn parse<'e>(&self,
        lineno: usize,
        metadata: Metadata<'e>,
        _rank: Option<u32>,
        compile_id: &Option<CompileId>,
        payload: &str
    ) -> anyhow::Result<ParseOutput> {
        if let Metadata::OptimizeDdpSplitChild(m) = metadata {
            let filename = format!("optimize_ddp_split_child_{}.txt", m.name);
            simple_file_output(&filename, lineno, compile_id, payload)
        } else {
            Err(anyhow::anyhow!("Expected OptimizeDdpSplitChild metadata"))
        }
    }
}

// Register your parser here
pub fn all_parsers<'t>(tt: &'t TinyTemplate<'t>) -> Vec<Box<dyn StructuredLogParser + 't>> {
    // We need to use Box wrappers here because vecs in Rust need to have known size
    let result : Vec<Box<dyn StructuredLogParser>> = vec![
        Box::new(SentinelFileParser::new("optimize_ddp_split_graph", |e| e.optimize_ddp_split_graph.as_ref())),
        Box::new(SentinelFileParser::new("compiled_autograd_graph", |e| e.compiled_autograd_graph.as_ref())),
        Box::new(SentinelFileParser::new("aot_forward_graph", |e| e.aot_forward_graph.as_ref())),
        Box::new(SentinelFileParser::new("aot_backward_graph", |e| e.aot_backward_graph.as_ref())),
        Box::new(SentinelFileParser::new("aot_joint_graph", |e| e.aot_joint_graph.as_ref())),
        Box::new(SentinelFileParser::new("inductor_post_grad_graph", |e| e.inductor_post_grad_graph.as_ref())),
        Box::new(DynamoOutputGraphParser),
        Box::new(DynamoGuardParser { tt }),
        Box::new(InductorOutputCodeParser),
        Box::new(OptimizeDdpSplitChildParser),
    ];

    result
}
