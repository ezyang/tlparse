use crate::types::*;
use std::path::PathBuf;

/**
 * StructuredLogParser
 * Parses a structured log and returns a vec of file outputs.
 * Implement this trait to add your own analyses.
 */
pub trait StructuredLogParser {
    // If this returns Some value, the parser will be run on that metadata.
    // Otherwise, it will be skipped. 'e is the lifetime of the envelope.
    fn get_metadata<'e>(&self, e: &'e Envelope) -> Option<Metadata<'e>>;
    fn parse<'e>(&self,
        lineno: usize, // Line number from log
        metadata: Metadata<'e>, // Metadata from get_metadata
        rank: Option<u32>, // Rank of the log
        compile_id: &Option<CompileId>, // Compile ID of the envelope
        payload: &str // Payload from the log (empty string when None)
    ) -> anyhow::Result<ParseOutput>;
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
        simple_file_output(self.filename, lineno, compile_id, payload)
    }
}

// Same as SentinelFileParser, but can log the size of the graph
pub struct DynamoOutputGraphParser;
impl StructuredLogParser for DynamoOutputGraphParser {
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

// Register your parser here
pub fn all_parsers() -> Vec<Box<dyn StructuredLogParser>> {
    // We need to use Box wrappers here because vecs in Rust need to have known size
    let result : Vec<Box<dyn StructuredLogParser>> = vec![
        Box::new(SentinelFileParser::new("optimize_ddp_split_graph.txt", |e| e.optimize_ddp_split_graph.as_ref())),
        Box::new(SentinelFileParser::new("compiled_autograd_graph.txt", |e| e.compiled_autograd_graph.as_ref())),
        Box::new(SentinelFileParser::new("aot_forward_graph.txt", |e| e.aot_forward_graph.as_ref())),
        Box::new(SentinelFileParser::new("aot_backward_graph.txt", |e| e.aot_backward_graph.as_ref())),
        Box::new(SentinelFileParser::new("aot_joint_graph.txt", |e| e.aot_joint_graph.as_ref())),
        Box::new(SentinelFileParser::new("inductor_post_grad_graph.txt", |e| e.inductor_post_grad_graph.as_ref())),
        Box::new(DynamoOutputGraphParser {}),
    ];
    result
}
