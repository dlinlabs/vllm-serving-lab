"""Versioned, precomputed replay inputs; no output-length oracle in requests."""
import hashlib
import json
import math
from pathlib import Path
import random

MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
MAX_MODEL_LEN = 8192


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def input_tokens(tokenizer, messages):
    ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                        add_generation_prompt=True, return_dict=True)['input_ids']
    if hasattr(ids, 'shape'):
        return int(ids.shape[-1])
    return len(ids[0]) if ids and isinstance(ids[0], (list, tuple)) else len(ids)


def tokenizer_fingerprint(tokenizer):
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    if backend is None:
        return None
    material = backend.to_str() + json.dumps(tokenizer.chat_template, sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()


def synthetic_prompt(tokenizer, target, key):
    """Generate varied text and verify the *full chat template* token count."""
    vocabulary = []
    for word in 'abcdefghijklmnopqrstuvwxyz':
        ids = tokenizer.encode(' ' + word, add_special_tokens=False)
        if len(ids) == 1:
            vocabulary.append(ids[0])
    if len(vocabulary) < 2:
        raise ValueError('Tokenizer has no suitable single-token synthetic vocabulary')
    empty = input_tokens(tokenizer, [{'role': 'user', 'content': ''}])
    rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16))
    sequence = [rng.choice(vocabulary) for _ in range(target + 16)]
    length = max(0, target - empty)
    for _ in range(12):
        if not 0 <= length <= len(sequence):
            break
        prompt = tokenizer.decode(sequence[:length], skip_special_tokens=True)
        actual = input_tokens(tokenizer, [{'role': 'user', 'content': prompt}])
        if actual == target:
            return prompt
        length += target - actual
    raise ValueError(f'Cannot synthesize exactly {target} chat-template tokens')


def finite_number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite numeric')
    if positive and value <= 0:
        raise ValueError(f'{name} must be positive')
    return value


def validate_trace(trace):
    if trace.get('schema_version') != 1 or trace.get('max_model_len') != MAX_MODEL_LEN:
        raise ValueError('Unsupported trace schema or context length')
    if not isinstance(trace.get('model'), str) or not trace['model']:
        raise ValueError('Trace model is missing')
    duration = finite_number(trace['duration_seconds'], 'duration', True)
    cap = trace['output_cap']
    if type(cap) is not int or not 0 < cap < MAX_MODEL_LEN:
        raise ValueError('Invalid output cap')
    requests = trace['requests']
    if not isinstance(requests, list) or not 0 < len(requests) <= 90000:
        raise ValueError('Trace needs 1..90000 requests')
    seen = set()
    previous = -1.0
    for row in requests:
        identifier = row['trace_id']
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError('Missing or duplicate trace ID')
        seen.add(identifier)
        arrival = finite_number(row['arrival_seconds'], 'arrival')
        if not 0 <= arrival < duration or arrival < previous:
            raise ValueError('Arrivals must be sorted and inside the measurement window')
        previous = arrival
        count = row['input_tokens']
        if type(count) is not int or count <= 0 or count + cap > MAX_MODEL_LEN:
            raise ValueError('Request exceeds context length or has invalid token count')
        if row['max_tokens'] != cap:
            raise ValueError('Every request must use the predeclared output cap')
        if set(row) != {'trace_id', 'arrival_seconds', 'input_tokens', 'max_tokens', 'prompt'}:
            raise ValueError('Unexpected replay request fields (no historical output lengths allowed)')
        if not isinstance(row['prompt'], str):
            raise ValueError('Prompt must be text')
    return trace


def load_trace(path):
    return validate_trace(json.loads(Path(path).read_text()))


def verify_prompts(trace, tokenizer):
    for row in trace['requests']:
        actual = input_tokens(tokenizer, [{'role': 'user', 'content': row['prompt']}])
        if actual != row['input_tokens']:
            raise ValueError(f"Tokenizer mismatch for {row['trace_id']}: {actual} != {row['input_tokens']}")
