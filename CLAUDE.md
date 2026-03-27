Follow these rules when working on the project

- When running Python scripts, always use `uv run`
- Never write down a summary markdown file after your work
- Always add new import statements to the top block of the file
- Do NOT do defensive programming -- it is better to let code fail than having defaults for everything and leave bugs uncaught
- When updating functions, do NOT consider backwards compatiability -- the code here is all for research and will not be used by others during development phase.

## When creating new task instances
- Always make sure the system prompt and evaluation methods align with what the original benchmark does. 
- The created dataset should have a `prompt` field that will be used by the harness.
- Provide a one-liner run command that can be used to test on the created task.