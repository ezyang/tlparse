pub static CSS: &str = r#"
table td { vertical-align: top; }

.stack-trie { white-space: nowrap; font-family: monospace; }
.stack-trie ul { padding-left: 1ch;  }
.stack-trie li { margin-left: 1ch; list-style-type: none;  }
.stack-trie .marker {
  cursor: pointer;
}
.stack-trie .marker.collapsed::before {
  content: "+ ";
}
.stack-trie .marker:not(.collapsed)::before {
  content: "- ";
}
.stack-trie a { text-decoration: none; }
.stack-trie a:hover { text-decoration: underline; }
.status-missing { background-color: purple; color: white; }
.status-error { background-color: red; color: white; }
.status-empty { background-color: white; color: black; }
.status-ok { background-color: green; color: white; }
.status-break { background-color: lime; color: black; }
"#;

pub static JAVASCRIPT: &str = r#"
  function toggleList(toggleItem) {
    const listItem = toggleItem.parentNode;
    const nestedList = listItem.querySelector('ul');
    if (nestedList) {
      nestedList.style.display = nestedList.style.display === 'none' ? 'block' : 'none';

      // Toggle the collapse/expand indicator
      toggleItem.classList.toggle('collapsed');
    }
  }
"#;

pub static TEMPLATE_DYNAMO_GUARDS: &str = r#"
<html>
<body>
<h2>Guards</h2>
<ul>
{{ for guard in guards }}
    <li><code>{guard.code}</code></li>
{{ endfor }}
</ul>
</body>
</html>
"#;

pub static TEMPLATE_INDEX: &str = r#"
<html>
<style>
{css | format_unescaped}
</style>
<script>
{javascript | format_unescaped}
</script>
<body>
<div>
{custom_header_html | format_unescaped}
<h2>Stack trie</h2>
<p>
The <strong>stack trie</strong> is a way of getting a quick orientation on where all the
compilations in a model take place, esp., if you are compiling a codebase you are unfamiliar with.
It is a tree of stack frames, for all stacks that triggered PT2 compilation.  If only a single
stack is in the tree, you will simply see a plain list of frames (most recent call last).  With
multiple stacks, at every point where two stacks diverge from having a common prefix, we increase
the indentation of the list and have a separate sub-list per sub-tree.
</p>
<p>
Links to particular compilation are color coded by status:
<span class="status-ok">[Success]</span>,
<span class="status-break">[Success with restart (e.g., graph break)]</span>,
<span class="status-empty">[Empty graph]</span>,
<span class="status-error">[Error]</span>,
<span class="status-missing">[Metrics were missing]</span>
</p>
{stack_trie_html | format_unescaped}
</div>
<div>
{{ if num_breaks }}
<h2> Failures and Restarts </h2>
<p>
Various issues may cause Dynamo to restart its analysis or give up on compilation entirely, causing graph breaks and fallbacks to eager mode.
This run had <strong><a href="failures_and_restarts.html">{num_breaks} restart(s) and/or compilation failure(s)</a></strong>.
</p>
{{ endif }}
<h2>IR dumps</h2>
<p>
The <strong>IR dumps</strong> collected dumped intermediate products from various points of the PT2
compilation process.  The products are organized by compile id, and then sorted in chronological
order.
</p>
<p>
A <strong>compile id</strong> uniquely identifies are particular compilation inside a PT2
program.  It is traditionally written as <code>[x/y]</code>, where the <strong>frame id</strong> x
identifies the particular Python frame which we are compiling, and <strong>frame compile
id</strong> y identifies how many times we've recompiled this same frame.  For example,
<code>[0/0]</code> refers to the very first frame compiled by PT2; <code>[0/1]</code> refers to the
first recompilation of this frame, while <code>[1/0]</code> refers to a different frame, within
distinct code cache, which we are compiling next (perhaps because of a graph break).  Although
Dynamo treats distinct frames as completely unrelated, a frame compilation could overlap with another
frame; for example, if you graph break in an inlined function, Dynamo will typically try to compile
the nested frame again on an inner frame.  You can identify the hierarchical relationship between
frames by looking at the stack trie above.
</p>
<p>
In some situations, the compile id will have an extra signifier <code>[x/y_z]</code>, where z is the
<strong>attempt</strong> for this particular (re)compilation.  Certain conditions will cause Dynamo to
restart analysis, when Dynamo discovers that it needs to undo a decision it previously made.  The most
common cause of recompilation is a graph break in an inlined function call, which forces to restart
and avoid inlining the function in the first place.
</p>
<p>
Here is a high level description of PT2's compilation phases, and the intermediate products each
phase generates:
</p>
<ol>
<li><em>Optional:</em> If compiled autograd is enabled, and we are processing a backward call, compiled autograd will trace the autograd graph from the autograd engine, and produce an FX graph <code>compiled_autograd_graph</code> that will be Dynamo traced.  Otherwise, Dynamo will directly trace user's bytecode.</li>
<li>Dynamo symbolically evaluates the Python bytecode of a program, producing <code>dynamo_output_graph</code></li>
<li><em>Optional:</em> If <code>optimize_ddp</code> is enabled, the DDPOptimizer will split the Dynamo output graph to improve pipelining communications.  Each split subgraph is <code>optimize_ddp_split_child_submod</code>, and the high level graph that plumbs the graphs together is <code>optimize_ddp_split_graph</code>.  If there are multiple splits, each subsequent build product will be produced multiple times, one for each split.</li>
<li>AOTAutograd traces the (possibly split) Dynamo output graph, producing a <code>aot_joint_graph</code> if backwards is enabled.  It then partitions the graph into <code>aot_forward_graph</code> and <code>aot_backward_graph</code>.  If training is not needed, there may only be an <code>aot_forward_graph</code>.</li>
<li>Inductor will apply some post grad FX passes, producing <code>inductor_post_grad_graph</code></li>
<li>Inductor will perform code generation, producing the final <code>inductor_output_code</code> which will be executed at runtime.  This output is a valid Python program and can be directly run.</li>
</ol>

{{ if has_chromium_events }}
<h2> Chromium Events </h2>
PT2 generates <a href='chromium_events.json'>Chromium Trace Events</a> in JSON on specific events during compilation.
You can download and view them in a tool like <a href='https://ui.perfetto.dev/'>Perfetto</a>.
{{ endif  }}
<p>
Build products below:
</p>
<ul>
{{ for compile_directory in directory }}
    <li><a id="{compile_directory.0}">{compile_directory.0}</a>
    <ul>
        {{ for path_idx in compile_directory.1 }}
            <li><a href="{path_idx.url}">{path_idx.name}</a> {path_idx.suffix} ({path_idx.number})</li>
        {{ endfor }}
    </ul>
    </li>
{{ endfor }}
</ul>
</div>

{{ if has_unknown_stack_trie }}
<div>
<h2>Unknown stacks</h2>
<p>
  Sometimes, logs are made without a compile id.  This makes it difficult to correlate related
  logs.  This stack trie shows all places where log entries occurred without compile context; to
  fix, look an appropriate place in the stack where compile id should have been specified.
</p>
{unknown_stack_trie_html | format_unescaped}
</div>
{{ endif }}
</body>
</html>
"#;

pub static TEMPLATE_FAILURES_CSS: &str = r#"
table {
    width: 90%;
    border-collapse: collapse;
    margin: 20px 0;
}
table, th, td {
    border: 1px solid #999;
    padding: 10px;
    text-align: left;
}
th {
    background-color: #d3d3d3;
    font-weight: bold;
}
tr:nth-child(odd) {
    background-color: #f2f2f2;
}
a {
    color: #0066cc;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}
"#;

pub static TEMPLATE_FAILURES_AND_RESTARTS: &str = r#"
<html>
<head>
    <style>
    {css}
    </style>
</head>
<body>
    <h1>Failures and Restarts</h1>
    <table>
    <tr> <th> Compile Id </th> <th> Failure Type </th> <th> Failure Description </th> <th> Failure Source (compilation failures only) </th> </tr>
    {{ for failure in failures }}
    <tr> <td> {failure.0 | format_unescaped} </td>{failure.1 | format_unescaped}</tr>
    {{ endfor }}
</body>
</html>
"#;

pub static TEMPLATE_COMPILATION_METRICS: &str = r#"
<html>
<head>
    <style>
    {css}
    </style>
    <title>Compilation Metrics</title>
    <base href="..">
</head>
<body>
    <h1>Compilation Info for {compile_id}</h1>
    <p>{mini_stack_html | format_unescaped}</p>
    <h2>Output files:</h2>
    <ul>
        {{ for path_idx in output_files }}
            <li><a href="{compile_id_dir}/{path_idx.url}">{path_idx.name}</a> ({path_idx.number})</li>
        {{ endfor }}
    </ul>
    <h2>Stack</h2>
    {stack_html | format_unescaped}
    <h2>Compile Time(seconds)</h2>
    <p>Entire Frame <abbr title="Total time spent in convert_frame function">[?]</abbr>: {m.entire_frame_compile_time_s}</div>
    <p>Backend <abbr title="Time spent running the backend compiler">[?]</abbr>: {m.backend_compile_time_s}</div>
    {{ if m.inductor_compile_time_s }}
    <p>Inductor <abbr title="Total time spent running inductor">[?]</abbr>: {m.inductor_compile_time_s}</div>
    {{ endif }}
    {{ if m.code_gen_time_s }}
    <p>Code Gen Time: {m.code_gen_time_s}</p>
    {{ endif}}
    <div>Dynamo Time Before Restart <abbr title="Total time spent restarting dynamo analysis">[?]</abbr>: {m.dynamo_time_before_restart_s}</div>
    <h2>Restarts and Failures</h2>
    {{ if m.fail_type }}
    <p>Failure Exception: <pre>{m.fail_type}</pre></p>
    <p>Failure Reason: <pre>{m.fail_reason}</pre></p>
    {{ if m.fail_user_frame_filename }}
    <p>In file <pre>{m.fail_user_frame_filename}</pre>, line {m.fail_user_frame_lineno}</p>
    {{ endif}}
    {{ else }}
    <p> No failures! </p>
    {{ endif }}
    {{ if m.restart_reasons }}
    <p>Restart Reasons:<p>
    {{ for restart_reason in m.restart_reasons }}
     <li> <code> {restart_reason} </code> </li>
    {{ endfor }}
    {{ else }}
    <p> No restarts! </p>
    {{ endif }}
    <h2>Cache Metrics</h2>
    <p>Cache Size: {m.cache_size}</p>
    <p>Accumulated Cache Size: {m.accumulated_cache_size}</p>
    <h2>Graph Metrics</h2>
    <p>Guard Count: {m.guard_count}</p>
    <p>Shape Env Guards: {m.shape_env_guard_count}</p>
    <p>Graph Ops: {m.graph_op_count}</p>
    <p>Graph Nodes: {m.graph_node_count}</p>
    <p>Graph Inputs: {m.graph_input_count}</p>
    <h2> Custom Ops </h2>
    <p> Compliant Custom Ops:</p>
    {{ for op in m.compliant_custom_ops }}
    <li> <code> {op} </code> </li>
    {{ endfor }}
    <p> Non-Compliant Custom Ops:</p>
    {{ for op in m.non_compliant_ops }}
    <li> <code> {op} </code> </li>
    {{ endfor }}
    <h2>Symbolic shape specializations</h2>
    <table>
    <tr>
        <th>Sym</th> <th>Source(s)</th> <th>Value</th> <th>User stack</th> <th>Framework stack</th>
    </tr>
    {{ for spec in symbolic_shape_specializations }}
    <tr>
        <td>{spec.symbol}</td>
        <td>{{ for source in spec.sources }}{source}<br>{{ endfor }}</td>
        <td>{spec.value}</td>
        <td>{spec.user_stack_html | format_unescaped}</td>
        <td>{spec.stack_html | format_unescaped}</td>
    </tr>
    {{ endfor }}
    </table>
</body>
</html>
"#;

pub static TEMPLATE_AOT_AUTOGRAD_BACKWARD_COMPILATION_METRICS: &str = r#"
<html>
<head>
    <style>
    {css}
    </style>
    <title>AOT Autograd Backward Compilation Metrics</title>
</head>
<body>
    <h1>Compilation Info for {compile_id}</h1>
    <h2>Failures</h2>
    {{ if m.fail_type }}
    <p>Failure Exception: <pre>{m.fail_type}</pre></p>
    <p>Failure Reason: <pre>{m.fail_reason}</pre></p>
    {{ else }}
    <p> No failures! </p>
    {{ endif }}
</body>
</html>
"#;

pub static TEMPLATE_BWD_COMPILATION_METRICS: &str = r#"
<html>
<head>
    <style>
    {css}
    </style>
    <title>Backward Compilation Metrics</title>
</head>
<body>
    <h1>Backward Compilation Info for {compile_id}</h1>
    <h2>Compile Time(seconds)</h2>
    {{ if m.inductor_compile_time_s }}
    <p>Inductor <abbr title="Total time spent running inductor">[?]</abbr>: {m.inductor_compile_time_s}</div>
    {{ endif }}
    {{ if m.code_gen_time_s }}
    <p>Code Gen Time: {m.code_gen_time_s}</p>
    {{ endif}}
    <h2>Failures</h2>
    {{ if m.fail_type }}
    <p>Failure Exception: <pre>{m.fail_type}</pre></p>
    <p>Failure Reason: <pre>{m.fail_reason}</pre></p>
    {{ else }}
    <p> No failures! </p>
    {{ endif }}
</body>
</html>
"#;
