# Human Annotation Guide

This fixed set contains 60 model responses: 30 from POPE and 30 from MathVista.

Fill these fields in `annotation_sheet.csv`:

- `human_is_hallucination`: `yes` or `no`. Mark yes when the model response is not supported by the image/question/reference answer.
- `human_hallucination_type`: choose one of `factual inconsistency`, `vision-grounding error`, `reasoning hallucination`, `none`, or `uncertain`.
- `human_evidence`: one short reason, e.g. `The image does not contain the queried object`, `The model counted incorrectly`, or `The final answer contradicts the chart`.
- `human_confidence_1_to_3`: `1` low, `2` medium, `3` high.

For POPE, `gold=no` and model answer `yes` is usually object hallucination / vision-grounding error.
For MathVista, an incorrect answer may be reasoning hallucination if visual information is read correctly but calculation or comparison is wrong; it may be vision-grounding error if the model misreads the image itself.
