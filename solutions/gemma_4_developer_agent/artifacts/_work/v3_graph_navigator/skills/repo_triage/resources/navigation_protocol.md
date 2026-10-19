# Navigation protocol

The graph tools are cheaper than reading files and much cheaper than being
wrong. Use them in this order.

| step | tool | why |
|---|---|---|
| 1 | `search_similar_code` | the embeddings find the symbol even when the issue never names it |
| 2 | `get_code_neighbors` | a bug is usually about an edge, not a node |
| 3 | `get_code_subgraph` | the induced subgraph is the contract you must not break |
| 4 | `read_file` with line ranges | read exactly the range the graph justified |

## Anti-patterns

- Grepping the whole repository first: costs turns, floods context, finds
  string matches rather than callers.
- Reading a 900-line module to change three lines.
- Editing before you can name the function that is wrong.
- Treating the test file as a hint about the fix: the test encodes the contract,
  it does not describe the implementation.
