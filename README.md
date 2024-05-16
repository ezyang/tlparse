# tlparse: Parse structured PT2 logs
`tlparse` parses structured torch trace logs and outputs HTML files analyzing data.

Quick start:
Run PT2 with the TORCH_TRACE environment variable set:
```
TORCH_TRACE=/tmp/my_traced_log example.py
```

Feed input into tlparse:
```
tlparse /tmp/my_traced_log -o tl_out/
```

# Adding custom parsers
You can extend tlparse with custom parsers which take existing structured log data and output any file. To do so, first implement StructuredLogParser with your own trait:

```Rust
pub struct MyCustomParser;
impl StructuredLogParser for MyCustomParser {
    fn name(&self) -> &'static str {
        "my_custom_parser"
    }
    fn get_metadata<'e>(&self, e: &'e Envelope) -> Option<Metadata<'e>> {
        // Get required metadata from the Envelope.
        // You'll need to update Envelope with your custom Metadata if you need new types here
        ....
    }

    fn parse<'e>(&self,
        lineno: usize,
        metadata: Metadata<'e>,
        _rank: Option<u32>,
        compile_id: &Option<CompileId>,
        payload: &str
    ) -> anyhow::Result<ParserResult> {
       // Use the metadata and payload however you'd like
       // Return either a ParserOutput::File(filename, payload) or ParserOutput::Link(name, url)
    }
}
```
