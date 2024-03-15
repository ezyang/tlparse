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
