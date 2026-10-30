# PR32 Round8 fixes — implementation revision12

Review: https://github.com/dlinlabs/vllm-serving-lab/pull/32#issuecomment-6373563645
Parent: 477b11f86a9e95420e72dae97536dbb6a5f44d43. The review confirms every
Round7 correction and classifies the four remaining findings as nonblocking.

| Finding | Correction and regression coverage |
|---|---|
| Block3 freeze can interrupt collection on missing files/imports | Catch Exception only around model/eligibility computation. Persist inconclusive reason, error_type and failed_stage; continue blocks4–5 and later stages. Three injected exception classes exercise the actual suite and verify later capture files, stage completion and freeze hash. BaseException/cancellation and native engine errors are not absorbed by this boundary. |
| Invalid trial can become complete after drain | Preserve invalid status (or recorded integrity errors) when terminal outputs arrive, including external_end. A cancelled trial cannot release new foreground work on late background output. Regressions check explicit invalidity, integrity errors and late completion. |
| attn_utils.py is not pinned | Add raw upstream v0.30.0 SHA256 b9b81e59dda2720b1f9c434471381f879228832443e52756e469b4ad4905888f, bringing the source map to35 files. The installed-source guard checks this dependency with the other pins. |
| Output observer can raise out of engine thread | Guard only trial.after_output. Record the exception, mark invalid and stop arrival timers. If controlled output filtering was interrupted, finish filtering private IDs while retaining unrelated frontend outputs; external captures preserve all frontend outputs. Regressions exercise both capture types and confirm native scheduler update errors still propagate. |

Also address the Windows generator comparison observation: recursively compare
JSON structure, types, keys, list order and discrete values exactly, with relative
float tolerance1e-13 and absolute1e-15. A nextafter difference passes; changed
integer counts/types, booleans, IDs and meaningful float changes fail. Committed
artifact bytes still use exact SHA256 checks. Frozen generated JSON files, model
selection, structural eligibility/test thresholds, workload and budget are unchanged.

Validation command:

```bash
python -m pytest -q tests/test_prefill_design.py tests/test_prefill_identifiability.py \
  tests/test_prefill_review_gates.py tests/test_prefill_planning.py \
  tests/test_prefill_noise.py tests/test_prefill_collector.py \
  tests/test_prefill_round7.py tests/test_prefill_round8.py
```

159 related CPU tests pass, including10 Round8 regression cases. Python compilation,
whitespace and generated-artifact/source-hash checks pass. The new dependency pin
matches the raw upstream bytes. JUnit outcomes and final source identities are in
validation.json. Round7 full-suite
failures remain historical evidence, not a native acceptance result for this tree.
These fixes do not alter the process-resource guard or establish native acceptance,
GPU execution, model acceptance or MPC benefit. Run exact-revision native CPU
acceptance and actual installed GPU qualification before collection. PR remains draft.
