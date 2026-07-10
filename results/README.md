# Result Provenance Policy

Formal Semantic-Motion results must record the input manifest, model, prompt
version, configuration, code revision, view mode, API failures and exclusions.

The historical `video2tasks_comparison_qwen36_paid_contact.json` result is
`prompt_only_legacy`: it used a contact sheet, a filename weak prior and a
closed task list for only the proposed prompt. Its 0.655 vs 0.195 composite
must not be reported as a full Video2Tasks pipeline comparison.

Likewise, rows containing semantic verifier fallback, mock verification, dummy
motion, missing paired views, or `pending_human` annotations are diagnostic
artifacts and are excluded from formal metrics.

Files prefixed `offline_` exercise the real LIBERO Fixed/Ego/EEF file path with
deterministic recognizer/mock components. They prove interface execution only,
not model accuracy.
