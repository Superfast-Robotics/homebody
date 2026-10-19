You keep the lessons file for an automated search that improves the built-in `{skill}` skill. Each
version below was written to fix a failure mechanism and then measured on a benchmark; its verdict
compares it with the best version of its time (value = mean of stage / 2 over episodes; "se" is the
standard error; "inert" means its change never ran).

Rewrite the lessons file from the whole history: at most 40 lines of Markdown, under three headings:
"## What helped" (changes that raised train or val value, even when not accepted, with version ids),
"## What failed and why" (changes that lowered value, were inert or were rejected, with the likely reason),
and "## Untried directions" (mechanisms the diagnoses named that no version has tried yet). Be concrete
(name the behaviour, not the version's prose), merge repeats, and keep earlier lessons that still hold.
Reply with the file's content only. Do not read or write any files.

<current lessons>
{lessons}
</current lessons>

<history>
{history}
</history>
