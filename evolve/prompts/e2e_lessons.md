You keep the lessons file for an automated search that improves the code agent sessions run on (profile {skill}: Dex1 skills and agent harness). Each
version below was written to fix a failure mechanism and then measured on a benchmark; its verdict
shows its change on the train sessions only, against the best version of its time ("se" is the standard
error). A version's probe record says whether its new skill code reported acting
(new_activations, activated_in) and in which train sessions the placed share changed; sessions vary from run to
run, so a change in a few sessions is weak evidence.

Rewrite the lessons file from the whole history: at most 40 lines of Markdown, under three headings:
"## What helped" (changes that raised the train placed share, with the measurement, its standard error and
whether activation was shown, even when not accepted, with version ids),
"## What failed and why" (changes that lowered the train placed share or were inert, with the likely reason;
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
