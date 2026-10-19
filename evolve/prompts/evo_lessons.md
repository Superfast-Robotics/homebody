You keep the lessons file for an automated search that improves the built-in `{skill}` skill. Each
version below was written to fix a failure mechanism and then measured on a benchmark; its verdict
compares it with the best version of its time on the train episodes only ("se" is the standard error; a
version rejected as "inert" reported no activation of its new code and changed no outcome on its probe
episodes; "probe ... activations" counts how often its new code reported acting).

Rewrite the lessons file from the whole history: at most 40 lines of Markdown, under three headings:
"## What helped" (changes that raised the train score, with the measurement, its standard error and
whether activation was shown, even when not accepted, with version ids),
"## What failed and why" (changes that lowered the train score or were inert, with the likely reason;
mark each reason as measured or guessed; a rejection for "no clear gain" is not a failed mechanism),
and "## Untried directions" (mechanisms the diagnoses named that no version has tried yet). Be concrete
(name the behaviour, not the version's prose), merge repeats, and keep earlier lessons that still hold.
Reply with the file's content only. Do not read or write any files.

<current lessons>
{lessons}
</current lessons>

<history>
{history}
</history>
