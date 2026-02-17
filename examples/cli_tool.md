# Data Format Converter CLI

Build a command-line tool for converting between data formats: CSV, JSON, YAML, and TOML.

## Features
- Convert between any pair of supported formats (CSV ↔ JSON ↔ YAML ↔ TOML)
- Read from file or stdin, write to file or stdout
- Schema inference for CSV (detect types: string, int, float, bool, date)
- Pretty-print or compact output modes
- Batch conversion (process a directory of files)
- Dry-run mode (show what would be converted without writing)

## CLI Interface
```
convert input.csv --to json --output result.json
convert input.json --to yaml                        # stdout
cat data.csv | convert --from csv --to toml         # stdin/stdout
convert ./data/ --to json --batch --output ./out/   # batch mode
convert input.csv --to json --dry-run               # preview
```

## Technical Requirements
- Python 3.12+ with Typer for CLI
- Handle nested structures (JSON/YAML/TOML → flattened CSV with dot notation)
- Handle arrays in CSV (comma-separated within cells or repeated rows)
- Streaming for large files (don't load entire file into memory)
- Clear error messages for malformed input
- Exit codes: 0 success, 1 conversion error, 2 invalid arguments

## Tests
- Round-trip conversion (CSV → JSON → CSV produces equivalent output)
- Nested data handling (JSON with nested objects → flat CSV)
- Type inference accuracy (dates, booleans, numbers)
- Stdin/stdout piping
- Invalid input error handling
- Batch mode with mixed file types

```bash
triad run --task "$(cat examples/cli_tool.md)"
```
