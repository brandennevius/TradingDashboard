# Strategy Knowledge + Trade Memory Plan

## Goal

Make AI trade reviews judge each trade against the actual strategy behind its tagged setup, not only the checklist score. A CANSLIM trade should be reviewed with CANSLIM context. An OTC trade should be reviewed with Branden's OTC trade plan context. Over time, the system should also learn from Branden's own best and worst historical examples.

## Principle

Do not “train” a model in a vague way. Use retrieval and structured examples:

- Setup Builder stores strategy knowledge for each setup.
- Trade reviews retrieve the knowledge for each trade's tagged setup.
- Future phases add curated historical examples marked by the trader.
- The LLM must cite/use only the strategy context, setup criteria, trade notes, executions, and selected past examples passed into the prompt.

## Phase 1 - Setup strategy knowledge

Status: Complete for manual/pasted knowledge sources.

Add a Strategy Knowledge section to each setup in Setup Builder.

Initial scope:

- Add knowledge sources to a setup.
- Each source can have:
  - title
  - type: Notes, Resource link, or Document excerpt
  - optional URL
  - pasted content / notes
- Persist this data with the existing setup checklist template.
- Include relevant setup knowledge in `Export review .docx`.
- If a trade is tagged `CANSLIM`, send CANSLIM knowledge to OpenAI for that trade.
- If a trade is tagged `OTC`, send OTC knowledge to OpenAI for that trade.

Out of scope for this first slice:

- Parsing uploaded PDFs/DOCX automatically.
- Embeddings/vector search.
- Visual screenshot interpretation.

## Phase 2 - Document/resource import

Add true import workflows for strategy knowledge.

Status: In progress. PDF/DOCX/TXT/Markdown upload now extracts text, chunks the document, stores chunks on the setup knowledge source, and review export retrieves the most relevant chunks per trade. Embeddings/vector search are still pending.

Planned scope:

- Upload PDF/DOCX/TXT/Markdown strategy documents.
- Extract text server-side.
- Split large documents into sections/chunks.
- Store chunks by setup knowledge source.
- Add active/inactive toggle so noisy or outdated sources can be kept but excluded from AI review.
- Add source labels and dates.
- Add active/inactive toggles so old strategy notes can be retired without deletion.
- For online resources, store URL plus a manual summary or fetched text where permitted.

Important copyright note:

- For CANSLIM, avoid storing copied book chapters or large copyrighted excerpts.
- Prefer Branden's own summarized rules, checklist notes, and permitted public notes/resources.

## Phase 3 - Strategy retrieval

Move from “send all setup notes” to retrieval.

Status: Started with lexical retrieval.

Completed:

- Score setup knowledge chunks against the trade's setup, notes, criteria, status, and executions.
- Send the top relevant chunks to OpenAI.
- Add a Strategy Sources Used section to each trade in the exported review.
- Ignore inactive knowledge sources during retrieval.

Still planned:

- Upgrade retrieval to embeddings/vector search if lexical matching is not good enough.
- Add better source summaries and active/inactive controls.

This keeps prompts smaller and prevents the review from being diluted by irrelevant strategy notes.

## Phase 4 - Trade memory / example library

Add a curated trade-memory layer.

Planned scope:

- On Trade Detail, allow marking a trade as:
  - Model example
  - Good execution
  - Bad execution
  - Avoid this pattern
  - Poor risk sizing
  - Poor entry timing
  - Strong management
  - Poor exit discipline
- Add a short “lesson learned” field.
- Store examples by setup.
- During review export, retrieve similar examples for the same setup.

The AI should then compare current trades to prior examples:

- “This resembles the prior AAPL CANSLIM winner because...”
- “This is closer to the failed HIMS example because...”

## Phase 5 - Smarter grading/review rubric

Use strategy knowledge + checklist + trade memory to generate richer review sections:

- Setup quality
- Entry quality
- Risk sizing quality
- Stop/invalidation quality
- Add/partial management
- Exit quality
- Comparison to model examples
- Comparison to avoid examples
- Next-week focus

The checklist remains the deterministic score. The LLM review becomes the qualitative coach layered on top.

## Phase 6 - Screenshot / chart vision

Status: Started.

Completed:

- Review export now sends a capped set of trade screenshots to OpenAI.
- The AI response must include a `chartAnalysis` field for each trade.
- The exported DOCX includes a Chart Analysis section under each trade.
- Image usage is capped to avoid excessive request size: up to 2 screenshots per trade and 10 screenshots per export.

Still planned:

- Better image compression/resizing before sending to OpenAI.
- UI control to choose which screenshots are included in AI review.
- Chart-specific checklist extraction, such as visible pivot, extension, moving-average status, and stop placement.

## Current implementation notes

- Setup criteria still drive grades.
- Strategy knowledge gives OpenAI deeper context for reviewing trades.
- OpenAI failure should fail the export instead of generating generic filler.
- The review export should only use currently filtered trades from Trade Log.
