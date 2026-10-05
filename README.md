# Verilog Agentic AI

A deployable full-stack agentic AI application that converts natural-language hardware requirements into Verilog HDL and an automatic testbench, then compiles/simulates the result and attempts automatic repair when verification fails.

## Architecture

1. **Requirement Understanding Agent** — converts plain English into a hardware specification and asks clarification questions when critical information is missing.
2. **RTL Generation Agent** — generates synthesizable Verilog-2001.
3. **Testbench Agent** — generates a self-checking Verilog testbench.
4. **Verification Tool** — runs Icarus Verilog compilation and simulation.
5. **Debug/Repair Agent** — receives compiler/simulation logs and repairs RTL/testbench for a bounded number of attempts.
6. **Final Report** — returns RTL, testbench, verification status and repair history.

## Local/Colab execution

Install:

```bash
pip install -r requirements.txt
apt-get update && apt-get install -y iverilog
```

Set `GEMINI_API_KEY`, then:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

## Render deployment

This project uses Docker so Icarus Verilog is installed automatically.

1. Push this folder to a GitHub repository.
2. In Render create a **Web Service** from the repository.
3. Choose **Docker** as the environment.
4. Add environment variable `GEMINI_API_KEY` with the key from Google AI Studio.
5. Optionally add `GEMINI_MODEL` if you want another model supported by your API account.
6. Deploy.

The frontend and backend are served from the same Render service, so the browser calls `/generate` directly. No hard-coded Render URL is required in the frontend.

## Example input

`Design a 2-to-1 multiplexer with inputs in0 and in1, select sel, and output out. When sel is 0 choose in0, otherwise choose in1.`

Expected successful response status: `VERIFICATION_PASS`.

## Important scope

The application generates **Verilog HDL**, not SystemVerilog. For requirements that omit essential information (for example, a counter with no width or reset behavior), the system asks a clarification question instead of silently inventing requirements.
