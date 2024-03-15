pub static CSS: &str = r#"
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
{css}
</style>
<body>
<div>
<h2>Stack trie</h2>
<p>
The <strong>stack trie</strong> is a way of getting a quick orientation on where all the
compilations in a model take place, esp., if you are compiling a codebase you are unfamiliar with.
It is a tree of stack frames, for all stacks that triggered PT2 compilation.  If only a single
stack is in the tree, you will simply see a plain list of frames (most recent call last).  With
multiple stacks, at every point where two stacks diverge from having a common prefix, we increase
the indentation of the list and have a separate sub-list per sub-tree.
</p>
{stack_trie_html | format_unescaped}
</div>
<div>
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
<p>
Build products below:
</p>
<ul>
{{ for compile_directory in directory }}
    <li><a id="{compile_directory.0}">{compile_directory.0}</a>
    <ul>
        {{ for path in compile_directory.1 }}
            <li><a href="{path}">{path}</a></li>
        {{ endfor }}
    </ul>
    </li>
{{ endfor }}
</ul>
</div>
</body>
</html>
"#;
