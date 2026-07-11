# Semantic-Motion Gold Annotation Protocol

Files under `annotations/` are candidate annotation packets, not gold labels.
Formal evaluation accepts only packets whose `annotation_status` is
`adjudicated` and which contain two distinct independent annotators.

## Annotation order

1. Verify that Fixed and Ego files show the same clock interval.
2. Mark atomic-task boundaries without reading model predictions.
3. Label macro domain/task, target, and destination.
4. Label 1–3 second micro actions with verb, object, body part, posture, and
   contact state. Use `unknown` when contact is not visible.
5. Record evidence frame indices separately for each view.
6. Assign `keep` only when video, motion, text, and synchronization are all
   valid. Record the concrete corruption type for every `drop`.
7. A second annotator repeats steps 1–6 independently. An adjudicator resolves
   disagreements and changes the status to `adjudicated`.

All boundary and segment times use the absolute clock of `source_sample_id`,
not a zero-based clock local to the candidate clip.

The LIBERO BDDL/file title is retained as `weak_task_title` for reference only.
It is not valid truth for boundaries, micro actions, contact, or keep/drop.
