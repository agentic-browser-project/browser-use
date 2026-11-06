# Raw HTML context length experiment

Final measurement: 450 fixed sampled WebVoyager steps, 425 retained paired observations, 308 retained tasks, 115 distinct requested URLs. This is a fresh URL revisit experiment, not an exact historical browser-state replay. Raw HTML was not stored in the offline trajectories.

## Source and protocol

- Source: `simulator/runs/WebVoyager-GAIA-topk32-30b-fixed-20260628`, 643 WebVoyager tasks and 5,871 recorded message contexts. This is a separate collection from Figure 1's 5,955-step dense run.
- Excluded before sampling: nine ambiguous/non-HTTP current-tab observations and four PDF observations, leaving 5,858 eligible steps.
- Uniform sampling without replacement within each task website, 30 steps per website, seed 20260922. Selected 323 tasks. No replacement of failed captures.
- All 66 shortened message URLs were restored from saved full URLs and checked against browser-use's 25-character query/fragment plus 7-digit MD5 alias algorithm.
- Fresh isolated headed Chrome 153 profiles, 1280 x 720 viewport, default extensions disabled, max_iframes=15, browser-use 0.12.1 at git commit 1bfae87f70c652c7e34fefb66de3e71cfa24f899.
- At least seven seconds after navigation, then up to 20 additional seconds waiting for document.readyState=complete. Dynamic pages are not assumed globally quiescent.
- Raw HTML: `document.documentElement.outerHTML` in the top-level document, including scripts, styles, hidden markup, and inline resource strings. Excludes child-frame documents, shadow-root interiors, external resource bodies, and the document type declaration.
- Fresh DOM: browser-use llm_representation with its default attributes and 40,000 character cap. None of the 425 accepted captures hit the cap.
- Only the user browser-state page payload is replaced. Original system/task/history/remaining text is retained. A historical truncated heading is normalized identically in both variants.
- Payloads are inserted verbatim without another URL-shortening pass. The source message collection has already undergone browser-use preprocessing. This explicit insertion convention is a length counterfactual, not an executed raw-HTML policy run.
- Qwen3-32B tokenizer, revision 9216db5781bf21249d130ec9da846c4624c16137, tokenizer SHA256 aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4. No tokenizer truncation/padding/special tokens.
- Each complete text part is retokenized, then counts are summed. Screenshot vision tokens, chat-template overhead, tool schema, and the current output are excluded. No model inference was run.
- Sampling was fixed before capture. The initial pilot exposed shortened URL aliases, was invalidated, and is retained separately under `raw_html_context_20260922_superseded_alias_pilot`. Only the corrected run contributes to the figure. The corrected run resumed from completed immutable results, with concurrency raised from 3 to 6.

## Final results

- Captures began 2026-09-22T22:12:21.894028+00:00 UTC and the last began 2026-09-22T22:33:09.303149+00:00 UTC.
- Exclusions: 15 HTTP errors, 8 DOM extraction failures, 1 navigation failure, and 1 stale-date Google Flights error. The last is an HTTP-200 content-QA exclusion. Original capture statuses are preserved separately from analysis statuses.
- Median raw context among retained steps: 225,169 tokens. Median fresh DOM context: 7,660 tokens.
- Median paired raw/DOM context ratio: 25.22851580. This differs from the ratio of pooled medians.
- Raw context range: 7,646 to 1,116,965 tokens.
- Maximum before/after HTML total-context token drift: 0.953709%. No observation was dropped based on this small drift.
- The sample allocation is website-balanced before capture failures, not benchmark-step-weighted. Results describe retained reconstructed contexts, not a population mean, task success, or runtime speedup.
- The Allrecipes and Booking samples each revisit one requested URL. Repeated URLs remain separate sampled steps with different frozen historical contexts. Session and page activity may also differ across fresh visits.
- Two Google Maps task samples visit Search Settings and show a cookies-disabled warning while the settings UI renders. An old Apple Watch product URL redirects to a current family page. These are retained as valid fresh pages; URL revisits do not restore historical cookies or page versions.

| Task website | Retained / attempted | Tasks | Requested URLs | DOM median | Raw median | Median paired ratio |
|---|---:|---:|---:|---:|---:|---:|
| Google Flights | 29/30 | 22 | 4 | 7,660.0 | 1,021,545.0 | 133.168 |
| Booking | 30/30 | 22 | 1 | 7,069.0 | 526,406.0 | 74.676 |
| Amazon | 30/30 | 20 | 8 | 8,220.5 | 458,884.0 | 55.929 |
| Coursera | 30/30 | 22 | 8 | 9,860.0 | 400,259.0 | 40.458 |
| GitHub | 30/30 | 19 | 11 | 7,101.0 | 307,666.0 | 43.064 |
| Allrecipes | 23/30 | 17 | 1 | 7,458.0 | 285,369.0 | 38.187 |
| Apple | 30/30 | 23 | 22 | 8,120.0 | 266,776.5 | 31.131 |
| Google Map | 30/30 | 23 | 15 | 7,158.0 | 231,925.0 | 30.378 |
| ESPN | 28/30 | 21 | 5 | 9,817.0 | 225,221.5 | 23.759 |
| BBC News | 29/30 | 18 | 11 | 7,367.0 | 149,068.0 | 20.570 |
| Google Search | 17/30 | 14 | 4 | 6,792.0 | 128,547.0 | 17.602 |
| Wolfram Alpha | 30/30 | 23 | 7 | 6,991.5 | 128,097.0 | 17.696 |
| Huggingface | 30/30 | 22 | 5 | 7,476.0 | 108,019.5 | 14.874 |
| Cambridge Dictionary | 30/30 | 20 | 5 | 7,374.5 | 96,143.0 | 13.119 |
| ArXiv | 29/30 | 22 | 9 | 10,211.0 | 17,627.0 | 1.663 |

## Reproduction and audit

From the browser-use repository:
```sh
uv run python simulator/reports/2026-09-22_raw_html_context/measure.py --run --workers 3
uv run python simulator/reports/2026-09-22_raw_html_context/audit_independent.py
```
Completed samples are reused, so the first command performs no additional visits after completion. To start a genuinely new capture collection, set a different output directory in the script and create its protocol with `--sample 30`; never overwrite this manifest.

From the paper repository:
```sh
python3 scripts/plot_raw_html_context.py
make paper
```

Local artifacts:
- `simulator/runs/raw_html_context_20260922/protocol.json`: immutable sample, exclusions, full URLs and hashes.
- `simulator/runs/raw_html_context_20260922/captures/<task>__step_NNN/`: raw_before/after.html.gz, dom_full.txt, dom_payload.txt, visible_text.txt, screenshot.jpg, result.json.
- `simulator/runs/raw_html_context_20260922/results.json`: all original capture outcomes.
- Paper `data/raw_html_context_20260922/measurements.csv` and `summary.json`: final curated analysis, including capture and analysis statuses.
- This directory's `independent_audit.json`: all 450 source/protocol hashes and 66 URL aliases verified; 17 independent reconstructed whole-message recounts across all 15 sites exactly match recorded/raw/DOM totals.
- `independent_review.md`: content QA. Real short login pages are retained. No new explicit bot walls or page failures were found in the final review.

Only the figure PDF, figure TeX wrapper, appendix TeX and its input statement are synchronized to Overleaf. Scripts, HTML, screenshots, CSV/JSON, logs, and preview files remain local.

Protocol SHA256: `3bf35e7f373e240b395e108fc639f1b66671de34d26d22f04aac67d6e25c8018`.
