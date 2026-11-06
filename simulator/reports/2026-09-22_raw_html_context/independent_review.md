# Independent review of raw HTML context experiment

Reviewed the v2 protocol with SHA-256 `3bf35e7f373e240b395e108fc639f1b66671de34d26d22f04aac67d6e25c8018` during collection and again after all 450 captures finished.

## Recount

`audit_independent.py` does not import the measurement script. It independently extracts the user-message browser payload using a regular expression and reconstructs each complete text content part before tokenization. The final deterministic recount included 17 observations covering all 15 websites. All recorded, raw HTML, and fresh DOM totals matched exactly. The JSON report lists individual checked observations and counts.

Source and protocol hashes passed for all 450 completed observations. All 66 shortened URLs in the fixed 450-step sample passed independent validation against the browser-use MD5 alias rule. The tokenizer checksum matched, and its truncation and padding settings were disabled.

## Failure review

- `Google_Flights--20`, step 005, is a missed application-level error despite HTTP 200. The visible page says that the requested flight date is in the past and reports no results. Flag this as a stale-date/application error, retain the capture in the audit, and exclude it from any primary distribution described as successful page retrievals. Do not change the dates in the recorded URL silently.
- `GitHub--10`, step 003, and `Wolfram_Alpha--24`, step 009, are valid short login-page observations. Their original and revisited URLs are already login URLs. They should remain in the sample.
- Small Google homepages and map pages are valid observations. Body length alone is not a failure criterion.
- Broad words such as “Verify” in BBC navigation, “Challenge” in Cambridge games, and “security verification codes” in Apple legal text are not error evidence.
- The final capture statuses are 426 captured, 15 HTTP errors, eight DOM extraction failures, and one navigation error. Excluding the one stale-date observation leaves 425 accepted pairs. No additional explicit HTTP 200 failure or bot wall was found in the final content review.
- Two Google Search Settings observations have a cookies-disabled warning but render the normal settings page. The original URL is already the settings URL. Apple Watch series 11 redirects to the broader Watch product page. Other inspected path redirects are ESPN regional canonicalization and Google Maps place/coordinate canonicalization. These remain valid fresh-page observations under the stated protocol.
- There are no recorded browser cleanup errors. No capture labelled `captured` has raw before/after total-text-token drift greater than 5 percent.

## Appendix text

The draft correctly states 5,871 recorded contexts, nine unavailable/ambiguous HTTP URLs, four PDF observations, 450 sampled steps from 323 tasks, 66 resolved aliases, and browser-use version 0.12.1. It distinguishes URL revisits from historical state recovery and states that freshly inserted payloads are not passed through URL shortening again.

When final results are available, add accepted versus attempted counts, the failure categories, the page-change check, and an explicit statement that the plotted distribution is conditional on accepted captures. These are important because failed search or stale-date pages are not a random subset of steps. The main document raw HTML excludes iframe and shadow interiors, while the serialized DOM may include them; the draft already makes this scope difference explicit.
