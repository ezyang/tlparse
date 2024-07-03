use crate::types::*;
use std::cell::RefCell;
use std::ffi::{OsStr, OsString};
use std::path::Path;
use std::path::PathBuf;
use tinytemplate::TinyTemplate;

use serde_json::Value;

use syntect::highlighting::ThemeSet;
use syntect::parsing::SyntaxSet;

pub enum ParserOutput {
    File(PathBuf, String), // File to be saved on disk
    Link(String, String), // External href to (name, url) (linked in compile_directory, not returned)
}

// Each parser returns a list of files to save and links to render in compile directory
pub type ParserResults = Vec<ParserOutput>;

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
    fn parse<'e>(
        &self,
        lineno: usize,                  // Line number from log
        metadata: Metadata<'e>,         // Metadata from get_metadata
        rank: Option<u32>,              // Rank of the log
        compile_id: &Option<CompileId>, // Compile ID of the envelope
        payload: &str,                  // Payload from the log (empty string when None)
    ) -> anyhow::Result<ParserResults>;

    // Name of the parser, for error logging
    fn name(&self) -> &'static str;
}

// Takes a filename and a payload and writes that payload into a the file
fn simple_file_output(
    filename: &str,
    lineno: usize,
    compile_id: &Option<CompileId>,
    payload: &str,
) -> anyhow::Result<ParserResults> {
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
    Ok(Vec::from([ParserOutput::File(f, String::from(payload))]))
}

/**
 * Parser for simple output dumps where the metadata is a sentinel {}
 */
pub struct SentinelFileParser {
    filename: &'static str,
    get_sentinel: fn(&Envelope) -> Option<&EmptyMetadata>,
}
impl SentinelFileParser {
    pub fn new(
        filename: &'static str,
        get_sentinel: fn(&Envelope) -> Option<&EmptyMetadata>,
    ) -> Self {
        Self {
            filename,
            get_sentinel,
        }
    }
}
impl StructuredLogParser for SentinelFileParser {
    fn name(&self) -> &'static str {
        self.filename
    }
    fn get_metadata<'e>(&self, e: &'e Envelope) -> Option<Metadata<'e>> {
        (self.get_sentinel)(e).map(|m| Metadata::Empty(m))
    }
    fn parse<'e>(
        &self,
        lineno: usize,
        _metadata: Metadata<'e>,
        _rank: Option<u32>,
        compile_id: &Option<CompileId>,
        payload: &str,
    ) -> anyhow::Result<ParserResults> {
        simple_file_output(
            &format!("{}.txt", self.filename),
            lineno,
            compile_id,
            payload,
        )
    }
}

/**
 * Generic parser for graph_dump entries
 */
pub struct GraphDumpParser;
impl StructuredLogParser for GraphDumpParser {
    fn name(&self) -> &'static str {
        "graph_dump" // ToDO: more specific?
    }
    fn get_metadata<'e>(&self, e: &'e Envelope) -> Option<Metadata<'e>> {
        e.graph_dump.as_ref().map(|m| Metadata::GraphDump(m))
    }
    fn parse<'e>(
        &self,
        lineno: usize,
        metadata: Metadata<'e>,
        _rank: Option<u32>,
        compile_id: &Option<CompileId>,
        payload: &str,
    ) -> anyhow::Result<ParserResults> {
        if let Metadata::GraphDump(metadata) = metadata {
            let filename: PathBuf = {
                let mut r = OsString::from(&metadata.name);
                r.push(OsStr::new(".txt"));
                r.into()
            };
            simple_file_output(&filename.to_string_lossy(), lineno, compile_id, payload)
        } else {
            Err(anyhow::anyhow!("Expected GraphDump metadata"))
        }
    }
}

// Same as SentinelFileParser, but can log the size of the graph
pub struct DynamoOutputGraphParser;
impl StructuredLogParser for DynamoOutputGraphParser {
    fn name(&self) -> &'static str {
        "dynamo_output_graph"
    }
    fn get_metadata<'e>(&self, e: &'e Envelope) -> Option<Metadata<'e>> {
        e.dynamo_output_graph
            .as_ref()
            .map(|m| Metadata::DynamoOutputGraph(m))
    }
    fn parse<'e>(
        &self,
        lineno: usize,
        _metadata: Metadata<'e>, // TODO: log size of graph
        _rank: Option<u32>,
        compile_id: &Option<CompileId>,
        payload: &str,
    ) -> anyhow::Result<ParserResults> {
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
    fn parse<'e>(
        &self,
        lineno: usize,
        _metadata: Metadata<'e>,
        _rank: Option<u32>,
        compile_id: &Option<CompileId>,
        payload: &str,
    ) -> anyhow::Result<ParserResults> {
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
        e.inductor_output_code
            .as_ref()
            .map(|m| Metadata::InductorOutputCode(m))
    }

    fn parse<'e>(
        &self,
        lineno: usize,
        metadata: Metadata<'e>,
        _rank: Option<u32>,
        compile_id: &Option<CompileId>,
        payload: &str,
    ) -> anyhow::Result<ParserResults> {
        if let Metadata::InductorOutputCode(metadata) = metadata {
            let filename = metadata
                .filename
                .as_ref()
                .and_then(|p| Path::file_stem(p))
                .map_or_else(
                    || PathBuf::from("inductor_output_code.html"),
                    |stem| {
                        let mut r = OsString::from("inductor_output_code_");
                        r.push(stem);
                        r.push(OsStr::new(".html"));
                        r.into()
                    },
                );
            // Generate HTML output and handle potential errors
            let html_payload = match generate_html_output(payload) {
                Ok(html) => html,
                Err(_e) => return Err(anyhow::anyhow!("Failed to parse inductor code to html")),
            };
            simple_file_output(
                &filename.to_string_lossy(),
                lineno,
                compile_id,
                &html_payload,
            )
        } else {
            Err(anyhow::anyhow!("Expected InductorOutputCode metadata"))
        }
    }
}

fn generate_html_output(payload: &str) -> Result<String, anyhow::Error> {
    let syntax_set = SyntaxSet::load_defaults_newlines();
    let theme_set = ThemeSet::load_defaults();
    let syntax = syntax_set.find_syntax_by_extension("py").unwrap();
    let html = syntect::html::highlighted_html_for_string(
        &payload,
        &syntax_set,
        &syntax,
        &theme_set.themes["InspiredGitHub"],
    );
    Ok(html?)
}

pub struct OptimizeDdpSplitChildParser;
impl StructuredLogParser for OptimizeDdpSplitChildParser {
    fn name(&self) -> &'static str {
        "optimize_ddp_split_child"
    }
    fn get_metadata<'e>(&self, e: &'e Envelope) -> Option<Metadata<'e>> {
        e.optimize_ddp_split_child
            .as_ref()
            .map(|m| Metadata::OptimizeDdpSplitChild(m))
    }

    fn parse<'e>(
        &self,
        lineno: usize,
        metadata: Metadata<'e>,
        _rank: Option<u32>,
        compile_id: &Option<CompileId>,
        payload: &str,
    ) -> anyhow::Result<ParserResults> {
        if let Metadata::OptimizeDdpSplitChild(m) = metadata {
            let filename = format!("optimize_ddp_split_child_{}.txt", m.name);
            simple_file_output(&filename, lineno, compile_id, payload)
        } else {
            Err(anyhow::anyhow!("Expected OptimizeDdpSplitChild metadata"))
        }
    }
}

pub struct LinkParser;
impl StructuredLogParser for LinkParser {
    fn name(&self) -> &'static str {
        "link_parser"
    }
    fn get_metadata<'e>(&self, e: &'e Envelope) -> Option<Metadata<'e>> {
        e.link.as_ref().map(|m| Metadata::Link(m))
    }

    fn parse<'e>(
        &self,
        _lineno: usize,
        metadata: Metadata<'e>,
        _rank: Option<u32>,
        _compile_id: &Option<CompileId>,
        _payload: &str,
    ) -> anyhow::Result<ParserResults> {
        if let Metadata::Link(m) = metadata {
            Ok(Vec::from([ParserOutput::Link(
                m.name.clone(),
                m.url.clone(),
            )]))
        } else {
            Err(anyhow::anyhow!("Expected Link Metadata"))
        }
    }
}

fn format_stack(stack: &StackSummary) -> String {
    let mut trie = StackTrieNode::default();
    trie.insert_no_terminal(stack.to_vec());
    trie.fmt(None).unwrap()
}

pub struct CompilationMetricsParser<'t> {
    pub tt: &'t TinyTemplate<'t>,
    pub stack_index: &'t RefCell<StackIndex>,
    pub symbolic_shape_specialization_index: &'t RefCell<SymbolicShapeSpecializationIndex>,
    pub output_files: &'t Vec<(String, String, i32)>,
}
impl StructuredLogParser for CompilationMetricsParser<'_> {
    fn name(&self) -> &'static str {
        "compilation_metrics"
    }
    fn get_metadata<'e>(&self, e: &'e Envelope) -> Option<Metadata<'e>> {
        e.compilation_metrics
            .as_ref()
            .map(|m| Metadata::CompilationMetrics(m))
    }
    fn parse<'e>(
        &self,
        lineno: usize,
        metrics: Metadata<'e>,
        _rank: Option<u32>,
        compile_id: &Option<CompileId>,
        _payload: &str,
    ) -> anyhow::Result<ParserResults> {
        let filename = format!("{}.html", self.name());
        if let Metadata::CompilationMetrics(m) = metrics {
            let id = compile_id
                .clone()
                .map_or("(unknown) ".to_string(), |c| format!("{cid} ", cid = c));
            let mut cid = compile_id.clone();
            if let Some(c) = cid.as_mut() {
                c.attempt = 0;
            }
            let stack_html = self
                .stack_index
                .borrow()
                .get(&cid)
                .map_or("".to_string(), format_stack);
            let specializations = self
                .symbolic_shape_specialization_index
                .borrow_mut()
                .remove(&cid)
                .unwrap_or(Vec::new())
                .drain(..)
                .map(|spec| SymbolicShapeSpecializationContext {
                    symbol: spec.symbol.unwrap_or("".to_string()),
                    sources: spec.sources.unwrap_or(Vec::new()),
                    value: spec.value.unwrap_or("".to_string()),
                    user_stack_html: format_stack(&spec.user_stack.unwrap_or(Vec::new())),
                    stack_html: format_stack(&spec.stack.unwrap_or(Vec::new())),
                })
                .collect();
            let remove_prefix = |x: &String| -> String {
                // url is X_Y_Z/<rest>. Get the rest of the string for the link
                // on compilation metrics page
                let parts: Vec<_> = x.split("/").collect();
                let new_str: String = parts[1..].join("");
                new_str
            };
            let output_files: Vec<(String, String, i32)> = self
                .output_files
                .iter()
                .map(|(url, name, number)| {
                    return (remove_prefix(url), remove_prefix(name), number.clone());
                })
                .collect();
            let context = CompilationMetricsContext {
                css: crate::CSS,
                m: &m,
                compile_id: id,
                stack_html: stack_html,
                symbolic_shape_specializations: specializations,
                output_files: &output_files,
            };
            let output = self.tt.render(&filename, &context)?;
            simple_file_output(&filename, lineno, compile_id, &output)
        } else {
            Err(anyhow::anyhow!("Expected CompilationMetrics metadata"))
        }
    }
}

pub struct AOTAutogradBackwardCompilationMetricsParser<'t> {
    tt: &'t TinyTemplate<'t>,
}
impl StructuredLogParser for AOTAutogradBackwardCompilationMetricsParser<'_> {
    fn name(&self) -> &'static str {
        "aot_autograd_backward_compilation_metrics"
    }
    fn get_metadata<'e>(&self, e: &'e Envelope) -> Option<Metadata<'e>> {
        e.aot_autograd_backward_compilation_metrics
            .as_ref()
            .map(|m| Metadata::AOTAutogradBackwardCompilationMetrics(m))
    }
    fn parse<'e>(
        &self,
        lineno: usize,
        metrics: Metadata<'e>,
        _rank: Option<u32>,
        compile_id: &Option<CompileId>,
        _payload: &str,
    ) -> anyhow::Result<ParserResults> {
        let filename = format!("{}.html", self.name());
        if let Metadata::AOTAutogradBackwardCompilationMetrics(m) = metrics {
            let id = compile_id
                .clone()
                .map_or("(unknown) ".to_string(), |c| format!("{cid} ", cid = c));
            let context = AOTAutogradBackwardCompilationMetricsContext {
                css: crate::CSS,
                m: &m,
                compile_id: id,
            };
            let output = self.tt.render(&filename, &context)?;
            simple_file_output(&filename, lineno, compile_id, &output)
        } else {
            Err(anyhow::anyhow!(
                "Expected AOTAutogradBackwardCompilationMetrics metadata"
            ))
        }
    }
}

pub struct ArtifactParser;
impl StructuredLogParser for ArtifactParser {
    fn name(&self) -> &'static str {
        "artifact"
    }
    fn get_metadata<'e>(&self, e: &'e Envelope) -> Option<Metadata<'e>> {
        e.artifact.as_ref().map(|m| Metadata::Artifact(m))
    }
    fn parse<'e>(
        &self,
        lineno: usize,
        metadata: Metadata<'e>,
        _rank: Option<u32>,
        compile_id: &Option<CompileId>,
        payload: &str,
    ) -> anyhow::Result<ParserResults> {
        if let Metadata::Artifact(metadata) = metadata {
            match metadata.encoding.as_str() {
                "string" => {
                    let filename = format!("{}.txt", metadata.name);
                    simple_file_output(&filename, lineno, compile_id, &payload)
                }
                "json" => {
                    let filename = format!("{}.json", metadata.name);
                    let value: Value = serde_json::from_str(&payload).unwrap();
                    let pretty = serde_json::to_string_pretty(&value).unwrap();
                    simple_file_output(&filename, lineno, compile_id, &pretty)
                }
                _ => Err(anyhow::anyhow!(
                    "Unsupported encoding: {}",
                    metadata.encoding
                )),
            }
        } else {
            Err(anyhow::anyhow!("Expected Artifact metadata"))
        }
    }
}

// Register your parser here
pub fn default_parsers<'t>(tt: &'t TinyTemplate<'t>) -> Vec<Box<dyn StructuredLogParser + 't>> {
    // We need to use Box wrappers here because vecs in Rust need to have known size
    let result: Vec<Box<dyn StructuredLogParser>> = vec![
        Box::new(SentinelFileParser::new("optimize_ddp_split_graph", |e| {
            e.optimize_ddp_split_graph.as_ref()
        })),
        Box::new(SentinelFileParser::new("compiled_autograd_graph", |e| {
            e.compiled_autograd_graph.as_ref()
        })),
        Box::new(SentinelFileParser::new("aot_forward_graph", |e| {
            e.aot_forward_graph.as_ref()
        })),
        Box::new(SentinelFileParser::new("aot_backward_graph", |e| {
            e.aot_backward_graph.as_ref()
        })),
        Box::new(SentinelFileParser::new("aot_joint_graph", |e| {
            e.aot_joint_graph.as_ref()
        })),
        Box::new(SentinelFileParser::new("inductor_post_grad_graph", |e| {
            e.inductor_post_grad_graph.as_ref()
        })),
        Box::new(SentinelFileParser::new("dynamo_cpp_guards_str", |e| {
            e.dynamo_cpp_guards_str.as_ref()
        })),
        Box::new(GraphDumpParser),
        Box::new(DynamoOutputGraphParser),
        Box::new(DynamoGuardParser { tt }),
        Box::new(InductorOutputCodeParser),
        Box::new(OptimizeDdpSplitChildParser),
        Box::new(AOTAutogradBackwardCompilationMetricsParser { tt }), // TODO: use own tt instances
        Box::new(LinkParser),
        Box::new(ArtifactParser),
    ];

    result
}
