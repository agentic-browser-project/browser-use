# Page32 continuation with reused legacy observations

Confirmed on 2026-11-13: return to page size 32 and maximum chunk size 16, retain A in {1,2,4,8,16,32,64,128} and alpha in {0.125,0.25,0.5}, and use action-type plus target-index agreement. The model remains Qwen3-32B BF16, with nominal 12.5% distinct-page budget, soft merge target 6, thinking enabled, greedy generation and a 1024-token response limit. Only spark00 is used.

## Sample and reuse

Count 64 distinct WebVoyager tasks. Preserve all 19 unique historical WebVoyager input states from 11 tasks, plus the two completed old-profile states from the subsequent run. Add the next 51 distinct tasks from the previously frozen candidate order. This gives 72 included states from 64 tasks. Three additional states run only as historical/fixed batch partners and never enter the dataset, task count, or rankings.

The old 21 states cover 20 settings. Fifteen overlap with the requested 24-setting grid. Retain those measured outputs and supplement the missing nine settings (A=2,64,128 at the three requested exponents). Five alpha=0.375 rows remain internal numerical controls only and are excluded from the reported grid. No page64/max32 outcome is reused as a page32/max16 measurement. Dense prompt KV may be reused when its exact prompt hash matches.

Preserve legacy model row positions and the original one- or two-state batch memberships. Legacy twenty-setting query pooling and score matrix shapes also remain unchanged. Use idle former cap32 rows for the nine added cap16 settings. The dense row remains row 40. Keep the original cache capacity for every state. Validate legacy token prefixes before new measurements. For each reused state, compare all twenty legacy cap16 continuations and the dense continuation against stored outputs before combining them with the nine additions. Any mismatch stops the merge for investigation. This costs additional inference but avoids silently combining incompatible execution schedules.

## Metrics and aggregation

Keep the existing strict parsing rule. A dense reference must finish and yield a parseable indexed action. A sparse output must finish, be parseable, and match both the entire action-type sequence and extracted target-index sequence. Invalid or unfinished sparse responses count as mismatches. Alternate XML or target-field spellings are not normalized in this metric. Consequently the score mixes formatting, completion and action fidelity. It is not pure semantic element accuracy and is not online task accuracy.

Average eligible steps within each task, then give tasks equal weight. Include a task in analysis only once every planned step for that task has a complete 24-setting grid. Retain tasks without eligible dense indexed actions in the inventory but exclude them from joint-metric denominators. Report type-only agreement, full-action equality, completion and parsing separately.

## Rolling exploratory selection

Start screening after 16 distinct tasks have complete grids, then update after every additional completed batch. Continue through all 64 tasks even if a favorable subset is found. Search subsets of exactly 16 and 32 eligible tasks.

The declared selection priority is lexicographic:

1. Maximize the worst-rival score margin of (32,0.25) against all other reported configurations.
2. At that optimum, minimize the worse rank of (16,0.25) and (8,0.25), then minimize their rank sum. Equal scores share competition ranks.
3. At those optima, maximize the (32,0.25) score.

Use integer-scaled task scores to preserve exact comparisons for tasks with multiple eligible steps. An optimal positive primary margin establishes unique first place, zero permits tied first place, and a negative optimum rules out first place within the measured pool for that subset size. A negative-margin optimum is not claimed to minimize the target's competition rank.

Every selection is explicitly outcome-dependent and favors named configurations. Preserve full-pool and frozen-prefix rankings, all measured outcomes, each intermediate selected set, and optimizer status. Do not interpret favorable selected-set ranks as unbiased superiority on unseen tasks.
