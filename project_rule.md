# Project Rules

This is a series study of a simple poker game

You will
- parse the instructions/**.md file when instructed,
- dump your understanding and plan in natural language into parsing/. Raise questions there if you need clarification
- then compose the simulation scripts/ and put results to artifacts/
- with key findings summarized into the tails of each instruction as well
- all with corresponding file/directory naming with the instruction index.


When an instruction ##.md is rewritten, move the corresponding outputs from ##.file or ##/ to ##/archive/ to update to the latest version
When an subinstruction ##-##.md is given, name the output accordingly

Instruction will cover the content to simulate, as well as visualization and expectations
You are expected to check the simulation output against the expectations
You are encouraged to log the results instead of relying solely on the visualization output to check the expectation


Future simulations will refer to existing instructions. You may be instructed to, or if you would, explicitly ask for permission to refactor the code to extract the common parts into common/ directory

You should always use the project virtual environment when running Python:

- **Local (macOS/Linux):** `~/Documents/projects/venv` (activate before tests/scripts).
- **GPU server (Windows, SSH `administrator@100.74.144.124`):** same layout under the user profile — **`C:\Users\Administrator\Documents\projects\venv`**. Use the interpreter **`C:\Users\Administrator\Documents\projects\venv\Scripts\python.exe`** (verified: Python 3.13.x).


You are allowed to execute the python scripts in place
You are encouraged to commit frequenctly with meaningful messages. You should often use commit --amend to keep meaningful commits only
You should keep the data file in artifacts/ folder to avoid resimulation each time.

**GPU server (mass rollout / heavy training):** SSH `administrator@100.74.144.124`. Repo checkout: `C:\Users\Administrator\Documents\projects\pick14`. Primary timed Q-training entrypoint: **`scripts/05_train_timed.py`**.

Example (from SSH — use **absolute** paths so `cmd`/`python` resolve the script; relative `scripts\...` can fail):

```
"C:\Users\Administrator\Documents\projects\venv\Scripts\python.exe" "C:\Users\Administrator\Documents\projects\pick14\scripts\05_train_timed.py" --games 800 --batch 64 --log-file train_run.log
```

Or run **`scripts/gpu_pickq_train.cmd`** via `cmd.exe /c C:\Users\Administrator\Documents\projects\pick14\scripts\gpu_pickq_train.cmd` (wrapper embeds those paths).

You should always use git to synchronize towards and from the server, including artifacts like training checkpoints.
