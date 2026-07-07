# CANSLIM Reference Library

This folder is the working index for your private chart-study material.

The source assets stay outside this folder:

- `../../Chart Setups/` contains the chart screenshots.
- `../../How to Make Money in Stocks - A Winning System in Good Times and Bad 4th edition 2009.pdf` contains the source book.

Keep the source assets private. The useful AI workflow is not model fine-tuning first; it is retrieval and structured review:

1. Index each chart image.
2. Tag the setup type and important visual cues.
3. Summarize the rule demonstrated by the example in your own words.
4. Compare new watchlist/trade screenshots against the tagged examples.
5. Produce a consistent setup review with a pass/watch/actionable decision.

## Files

- `chart_setup_index.csv`: one row per screenshot, with blank fields ready for tagging.
- `setup_review_schema.md`: the checklist the AI should use when reviewing your setups.

## Labeling Priorities

Start with these fields before trying to make the dataset exhaustive:

- `setup_type`
- `ticker`
- `timeframe`
- `base_quality`
- `volume_notes`
- `relative_strength_notes`
- `buy_point_notes`
- `failure_warnings`
- `model_lesson`

Once 20-30 examples are tagged well, the assistant can start comparing your own setups to the model examples in a useful way.
