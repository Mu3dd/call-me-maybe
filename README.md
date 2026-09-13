*This project has been created as part of the 42 curriculum by <your_login>.*

# call me maybe — Function Calling with Constrained Decoding

## Description

This project turns natural-language prompts (e.g. *"What is the sum of 40 and 2?"*)
into structured function calls (e.g. `{"name": "fn_add_numbers", "parameters": {"a": 40, "b": 2}}`)
using a small (0.6B parameter) language model.

The interesting part isn't calling an LLM — it's guaranteeing the LLM's output is
always **valid, schema-compliant JSON**, even though small models are notoriously
unreliable at following format instructions on their own. This is solved with
**constrained decoding**: instead of trusting the model to spontaneously produce
correct JSON, the program inspects the model's raw output probabilities (logits)
at every single generation step and forbids any token that would break the JSON
structure or violate the function's parameter schema.

## Instructions

### Requirements
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management

### Install
```bash
make install
# equivalent to: uv sync
```

### Run
```bash
make run
# equivalent to: uv run python -m src
```

By default this reads `data/input/functions_definition.json` and
`data/input/function_calling_tests.json`, and writes
`data/output/function_calling_results.json`. Custom paths:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

### Lint
```bash
make lint          # flake8 + mypy (required flags)
make lint-strict    # flake8 + mypy --strict
```

### Debug
```bash
make debug          # runs the program under pdb
```

## About `llm_sdk/` in this repository

The real `llm_sdk` package (wrapping the actual Qwen/Qwen3-0.6B model) is provided
separately by the school and must be copied into this repository, next to `src/`,
before real grading. **The `llm_sdk/` folder currently in this repo is a
hand-written mock**, built so the constrained-decoding logic could be developed
and tested without needing the real model weights. It implements the exact same
public interface (`encode`, `decode`, `get_logits_from_input_ids`,
`get_path_to_vocab_file`), but its "logits" are pseudo-random numbers, not a
trained model's real predictions — see the docstring at the top of
`llm_sdk/small_llm_model.py` for the full explanation of what this does and does
not simulate faithfully. Swapping in the real `llm_sdk` requires no code changes
in `src/`, only replacing the `llm_sdk/` folder — assuming its vocab file uses the
same array format the mock uses (see "Design decisions" below for why this
assumption might need revisiting).

## Resources

- [OpenAI function calling documentation](https://platform.openai.com/docs/guides/function-calling) — the general concept this project reimplements from scratch.
- [Outlines library](https://github.com/dottxt-ai/outlines) — a real-world constrained/structured decoding library; useful for comparing design choices against a production implementation.
- [Hugging Face — "How do Transformers work?"](https://huggingface.co/learn/nlp-course/chapter1/4) — background on tokenization, logits, and autoregressive generation.
- Andrej Karpathy, *"Let's build the GPT Tokenizer"* (YouTube) — explains subword tokenization and vocab files in depth, relevant to `src/vocab.py`.

**How AI was used:** An AI assistant (Claude) was used to help design and scaffold
this project — proposing the module split (`models.py` / `io_utils.py` / `vocab.py`
/ `constrained_decoder.py` / `resolver.py`), writing an initial implementation of
the token-masking logic, and drafting this README's structure. All generated code
was read, tested (see "Testing strategy" below), and understood line by line before
being kept; the mock `llm_sdk` in particular was reasoned through manually to make
sure its simplifications were deliberate and documented rather than hidden.

## Algorithm explanation — constrained decoding

Every generation step in this project goes through one function:
`constrained_step(sdk, vocab, context_ids, allowed_tokens)` in
`src/constrained_decoder.py`. It:

1. Calls `sdk.get_logits_from_input_ids(context_ids)` to get a score for every
   token in the vocabulary.
2. Builds a mask: every token *not* in `allowed_tokens` gets its score set to
   `-infinity`.
3. Picks the highest-scoring remaining token (`argmax` over the masked scores).
4. Appends that token's id to the running context and returns its string.

The rest of the file is just different ways of computing `allowed_tokens` at each
position in the target JSON:

- **Forced/literal characters** (`emit_literal`): `allowed_tokens` is a single
  character — e.g. after `{"name": "fn_add_numbers`, the only legal next
  character is `"`. There's no real "choice" here, but routing it through the
  same masked-logit mechanism keeps the whole pipeline uniform and means every
  character of the output — not just the "interesting" ones — is guaranteed valid.
- **Function name choice** (`choose_function_name`): `allowed_tokens` is the set
  of all known function names. Whichever one the (masked) model scores highest
  is chosen — in one step, since each function name is a single vocabulary token
  in the mock SDK.
- **Number fields** (`generate_number_field`): at each step, `allowed_tokens` is
  digits `0`-`9`, plus `-` only as the very first character, plus `.` only once
  and only after at least one digit, plus the field's terminator character
  (`,` or `}`) once at least one digit exists. Choosing the terminator ends the
  field — this means "when to stop" is itself a masked decision, not a hardcoded
  length.
- **String fields** (`generate_string_field`): opens with a forced `"`, then
  `allowed_tokens` is letters + space, plus the closing `"` once at least one
  character has been written, then the terminator.
- **Boolean fields** (`generate_boolean_field`): a single forced choice between
  the whole tokens `"true"` and `"false"`.

`generate_parameters` walks a function's declared parameters in schema order,
emitting each key as a forced literal and delegating to the right `generate_*`
function based on the parameter's declared type.

## Design decisions

- **One primitive, not many code paths.** Rather than writing separate logic for
  "generate forced text" vs. "generate a real choice," everything funnels through
  `constrained_step`, differing only in the size of the allowed set. This made the
  code far easier to test and to explain, at a small cost in efficiency (a forced
  character still costs one model call, even though it has only one legal option).
- **Deterministic keys/structure are handled the same way as free-form fields.**
  This was chosen for faithfulness to "every token passes through logit masking",
  matching the subject's description, over the alternative (skipping model calls
  entirely for parts with no real choice), which would technically be faster but
  less literal.
- **The vocab file format is assumed to be a flat JSON array of strings**
  (id = index). This is documented explicitly in `src/vocab.py` and
  `llm_sdk/small_llm_model.py` as an assumption that must be re-checked against
  the real `llm_sdk`'s actual vocab file before trusting this code unmodified —
  a real subword tokenizer's vocab file might be a `{token: id}` dict instead, or
  might split a function name across multiple tokens rather than giving it one
  whole token (see "Challenges faced").
- **Failures are isolated per prompt**, not per run. `_resolve_all` in
  `src/__main__.py` catches exceptions from `resolve_prompt` individually, logs a
  warning to stderr, and continues — so one malformed or unresolvable prompt
  doesn't take down the entire batch or leave `data/output/` empty.
- **A single custom exception type per module boundary**
  (`InputLoadError`, `VocabFormatError`, `ConstrainedDecodingError`) rather than
  letting `FileNotFoundError`, `json.JSONDecodeError`, `pydantic.ValidationError`,
  etc. propagate directly. This keeps `try/except` blocks in `__main__.py` simple
  and guarantees no unhandled exception type can crash the program.

## Performance analysis

- **Validity: 100%.** Because every character is chosen from a pre-computed
  legal set (never freely sampled), the output is guaranteed to be parseable
  JSON matching the function's schema by construction — this was verified by
  parsing every entry of `function_calling_results.json` with `json.loads` and
  checking its keys match the schema exactly (see "Testing strategy").
- **Value accuracy: not evaluated for the mock.** The mock `llm_sdk` has no real
  language understanding (see the note above), so it cannot be expected to
  extract the *correct* numbers/names from a prompt — only a real trained model
  (via the real `llm_sdk`) can be meaningfully measured against the subject's
  90%+ accuracy target. The 100% validity guarantee, however, holds regardless
  of which model is plugged in, since it's enforced by the masking logic, not by
  the model's competence.
- **Speed:** each generated character costs exactly one `get_logits_from_input_ids`
  call. For the sample data (5 prompts, 1–2 parameters each), this runs in well
  under a second with the mock SDK; a real model's inference cost per call would
  dominate real-world runtime, but the number of calls per prompt stays small
  (roughly: 1 call for the function name + a handful of calls per character of
  each argument value).

## Challenges faced

- **Deciding when a free-form field should stop.** Digits and string content
  don't have a fixed length, so "stop" had to be modeled as one of the legal
  choices at each step (the field's own terminator character) rather than a
  separate mechanism — see `generate_number_field`/`generate_string_field`.
- **Type-checking a value that's genuinely one of three different Python types.**
  `generate_parameters` produces a `float`, `str`, or `bool` depending on the
  parameter's declared type; mypy could not infer a single type across the
  `if/elif` branches, so `value` and `field_text` needed explicit union type
  annotations before the branches.
- **The mock's whole-token function names won't generalize to a real subword
  tokenizer.** In this mock, `"fn_add_numbers"` is one vocabulary entry, so
  choosing a function is a single masked decision. A real tokenizer would very
  likely split that name into several sub-word tokens, meaning
  `choose_function_name` would need to become a multi-step, trie-based
  prefix-matching process instead of a single call — this is flagged directly in
  `constrained_decoder.py`'s error message if a required token isn't found whole
  in the vocabulary.

## Testing strategy

Since automated tests aren't part of the graded submission, this project was
validated by:

1. Running `make run` on the shipped example data and manually inspecting
   `data/output/function_calling_results.json`.
2. Parsing every output entry with `json.loads` and asserting its keys are
   exactly `{prompt, name, parameters}`, to confirm structural validity
   independent of value correctness.
3. Deliberately breaking each input file (missing file, invalid JSON syntax,
   an out-of-schema value like `"type": "not_a_real_type"`) and confirming the
   program prints a clear message to stderr and exits with code 1, instead of
   crashing with a raw traceback.
4. Running with an empty prompts list (`[]`) to confirm the program still exits
   cleanly and writes an empty `[]` results file rather than erroring.
5. Running `make lint` (flake8 + the exact required mypy flags) and confirming
   zero errors.

## Example usage

```bash
make install
make run
cat data/output/function_calling_results.json
```

Expected shape of each entry (values will vary — see the accuracy note above):
```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```
