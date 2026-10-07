# LLM judge on the annotation sample

`sample_llm_judge.csv` holds the same 200 comments as `sample_for_annotation.csv`,
labelled by an LLM rather than a person. It is kept apart from the human sheet
(`sample_annotated.csv`) and scored and reported separately.

* Judge: Claude (configured model `claude-opus-5-5`), in a chat session on
  7 October 2026; zero-shot, one pass, no examples.
* Input: the comment text only, exactly as in the sheet. The judge did not see
  the distant labels, the label source, the subreddit or the key file.
* Task, per comment: overall sentiment of the writer (`pos`, `neg`, `neutral`),
  and whether a Nigerian or a US English writer is more likely (`NG`, `US`,
  `unsure` when the text gives no usable cue). Any cue in the text was allowed,
  including topic words, so this is a text-level reference point comparable to
  the `tokens` variant, not to `masked`.
* Scored with `python src/annotation.py score annotation/sample_llm_judge.csv`.

Limits: one judge, one run, not repeatable through an API from this repository.
The judge's variety guesses measure what the raw text reveals to a capable
reader; the post-hoc attacker measures what survives inside the encoder.
