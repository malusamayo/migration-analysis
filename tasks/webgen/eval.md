## Navigation Instruction
Evaluate a website by navigating through the websites and interacting with different elements extensively.

Key Guidelines You MUST follow:
* Action guidelines *
1) Use `browser_get_content` to extract the main content of the current page in clean markdown format.
2) Use `browser_get_state` to get the current state of the page. This provides you a json object of different elements you can interact with.
3) NEVER stop early. You need to fully explore the website to provide reliable evaluations. Test different functionalities user requested.

* Judgement guidelines *
When asked for a final judgement, you must respond with a JSON block in the following format:
```json
{"pass": true/false, "reason": "brief explanation"}
```
