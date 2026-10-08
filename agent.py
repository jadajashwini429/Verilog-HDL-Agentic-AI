import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Tuple

try:
    from google import genai
except ImportError:
    genai = None


MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
MAX_REPAIR_ATTEMPTS = int(os.getenv("MAX_REPAIR_ATTEMPTS", "2"))
IVERILOG = os.getenv("IVERILOG_PATH", "iverilog")
VVP = os.getenv("VVP_PATH", "vvp")


@dataclass
class AgentResult:
    verification_status: str
    rtl_code: str | None
    testbench_code: str | None
    final_report: Dict[str, Any]
    hardware_spec: Dict[str, Any] | None = None
    attempts: int = 0


class VerilogAgent:
    """Multi-stage Verilog generation and verification agent.

    Stages:
      1. Requirement understanding / specification
      2. RTL generation
      3. Testbench generation
      4. Compilation and simulation
      5. Automatic repair of RTL/testbench
      6. Final report
    """

    def __init__(self, api_key: str | None = None):
        key = api_key or os.getenv("GEMINI_API_KEY")

        if not key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")

        if genai is None:
            raise RuntimeError("google-genai is not installed.")

        self.client = genai.Client(api_key=key)

    def llm(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        retries: int = 3,
    ) -> str:
        """Call the LLM with retry handling for temporary API failures."""

        prompt = (
            f"SYSTEM ROLE:\n{system}\n\n"
            f"USER INPUT:\n{user}\n\n"
            "Return only the requested output. "
            "Do not add explanations unless explicitly requested."
        )

        last_error = None

        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                    config={"temperature": temperature},
                )

                text = getattr(response, "text", None)

                if not text:
                    raise RuntimeError("The model returned an empty response.")

                return text.strip()

            except Exception as exc:
                last_error = exc
                error_text = str(exc).lower()

                temporary_error = any(
                    keyword in error_text
                    for keyword in [
                        "503",
                        "unavailable",
                        "429",
                        "resource exhausted",
                        "high demand",
                        "temporarily",
                    ]
                )

                if not temporary_error or attempt == retries - 1:
                    raise

                wait_time = 2 ** attempt
                time.sleep(wait_time)

        raise RuntimeError(f"LLM request failed: {last_error}")

    @staticmethod
    def clean_code(text: str, language: str = "verilog") -> str:
        """Remove accidental Markdown code fences from generated code."""

        text = text.strip()

        text = re.sub(
            r"^```(?:verilog|systemverilog|v)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        return text.strip()

    @staticmethod
    def parse_json(text: str) -> Dict[str, Any]:
        """Parse JSON even if the model accidentally adds code fences."""

        text = text.strip()

        if text.startswith("```"):
            text = re.sub(
                r"^```(?:json)?\s*",
                "",
                text,
                flags=re.IGNORECASE,
            )
            text = re.sub(r"\s*```$", "", text)

        try:
            return json.loads(text)

        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)

            if match:
                return json.loads(match.group(0))

            raise

    def understand(self, request: str) -> Dict[str, Any]:
        """Convert the user's natural-language request into a hardware specification."""

        system = """You are the Requirement Understanding Agent for a Verilog HDL automation system.

Convert natural-language hardware requirements into a precise implementable specification.

If critical information is missing, do NOT invent it.
Set clarification_needed=true and provide concise questions.

If enough information exists, infer only conventional details and set clarification_needed=false.

Target Verilog-2001/Verilog HDL, not SystemVerilog.

Return JSON only with exactly these fields:

{
  "clarification_needed": boolean,
  "clarification_questions": [string],
  "module_name": string,
  "description": string,
  "parameters": [{"name": string, "value": string}],
  "inputs": [{"name": string, "width": integer, "description": string}],
  "outputs": [{"name": string, "width": integer, "description": string}],
  "behavior": [string],
  "clocking": string,
  "reset": string,
  "assumptions": [string]
}"""

        spec = self.parse_json(
            self.llm(
                system,
                request,
                temperature=0.1,
            )
        )

        spec.setdefault("parameters", [])
        spec.setdefault("inputs", [])
        spec.setdefault("outputs", [])
        spec.setdefault("behavior", [])
        spec.setdefault("assumptions", [])
        spec.setdefault("clarification_questions", [])

        return spec

    def generate_rtl(self, spec: Dict[str, Any]) -> str:
        """Generate synthesizable Verilog RTL."""

        system = """You are the RTL Generation Agent.

Generate synthesizable Verilog HDL only.

Rules:
- Use Verilog HDL (Verilog-2001), not SystemVerilog.
- Generate exactly one top-level module using the specified module_name.
- Match every port name and width in the specification exactly.
- No testbench.
- No markdown.
- No explanation outside the code.
- Prefer simple synthesizable constructs.
- For combinational logic use assign or always @*.
- For sequential logic use always @(posedge clock) and the specified reset behavior.
- Do not create undeclared ports or signals.
- Finish with endmodule."""

        return self.clean_code(
            self.llm(
                system,
                json.dumps(spec, indent=2),
                temperature=0.15,
            )
        )

    def generate_testbench(
        self,
        spec: Dict[str, Any],
        rtl: str,
    ) -> str:
        """Generate a Verilog-2001 testbench for the generated RTL."""

        system = """You are the Verification/Testbench Agent for Verilog HDL.

Generate a self-contained Verilog-2001 testbench for the supplied DUT.

Rules:
- Testbench module must be named tb_<dut_module_name>.
- Instantiate the DUT using exactly its declared ports.
- Exercise representative normal cases and important boundary/corner cases.
- For combinational circuits, use deterministic checks with if statements and $display.
- For sequential circuits, generate the required clock/reset and allow enough simulation time.
- The testbench must print exactly VERIFICATION_PASS if all checks pass.
- The testbench must print VERIFICATION_FAIL if any check fails.
- End with $finish.
- Do not use SystemVerilog-only features.
- Return code only."""

        payload = json.dumps(
            {
                "spec": spec,
                "rtl": rtl,
            },
            indent=2,
        )

        return self.clean_code(
            self.llm(
                system,
                payload,
                temperature=0.2,
            )
        )

    def run_tools(
        self,
        rtl: str,
        tb: str,
    ) -> Tuple[bool, str, str]:
        """Compile and simulate the generated Verilog in an isolated workspace."""

        with tempfile.TemporaryDirectory(
            prefix="verilog_agent_"
        ) as td:

            root = Path(td)

            rtl_path = root / "design.v"
            tb_path = root / "testbench.v"
            out_path = root / "sim.out"

            rtl_path.write_text(
                rtl,
                encoding="utf-8",
            )

            tb_path.write_text(
                tb,
                encoding="utf-8",
            )

            compile_proc = subprocess.run(
                [
                    IVERILOG,
                    "-g2005-s",
                    "-o",
                    str(out_path),
                    str(rtl_path),
                    str(tb_path),
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )

            compile_log = (
                compile_proc.stdout
                + "\n"
                + compile_proc.stderr
            ).strip()

            if compile_proc.returncode != 0:
                return (
                    False,
                    "COMPILE_ERROR",
                    compile_log,
                )

            sim_proc = subprocess.run(
                [
                    VVP,
                    str(out_path),
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )

            sim_log = (
                sim_proc.stdout
                + "\n"
                + sim_proc.stderr
            ).strip()

            if sim_proc.returncode != 0:
                return (
                    False,
                    "SIMULATION_ERROR",
                    sim_log,
                )

            if "VERIFICATION_PASS" not in sim_log:
                return (
                    False,
                    "VERIFICATION_FAIL",
                    sim_log,
                )

            return (
                True,
                "VERIFICATION_PASS",
                sim_log,
            )

    def repair(
        self,
        spec: Dict[str, Any],
        rtl: str,
        tb: str,
        stage: str,
        log: str,
    ) -> Tuple[str, str]:
        """Repair RTL and/or testbench after a verification failure."""

        system = """You are the Debug and Repair Agent for a Verilog generation pipeline.

A generated design/testbench failed compilation or simulation.

Analyze the failure and return JSON only:

{
  "rtl_code": "...",
  "testbench_code": "..."
}

Rules:
- Preserve the user's intended behavior.
- Fix only what is necessary.
- Keep Verilog-2001 compatibility.
- Ensure the DUT/testbench interfaces agree exactly.
- Ensure the testbench prints VERIFICATION_PASS on success.
- Ensure the testbench prints VERIFICATION_FAIL on failure.
- Never put markdown fences around code inside the JSON strings."""

        payload = json.dumps(
            {
                "spec": spec,
                "failure_stage": stage,
                "failure_log": log,
                "rtl_code": rtl,
                "testbench_code": tb,
            },
            indent=2,
        )

        result = self.parse_json(
            self.llm(
                system,
                payload,
                temperature=0.1,
            )
        )

        repaired_rtl = result.get(
            "rtl_code",
            rtl,
        )

        repaired_tb = result.get(
            "testbench_code",
            tb,
        )

        return (
            self.clean_code(repaired_rtl),
            self.clean_code(repaired_tb),
        )

    def run(self, request: str) -> AgentResult:
        """Run the complete Verilog generation and verification pipeline."""

        spec = self.understand(request)

        if spec.get("clarification_needed"):
            return AgentResult(
                verification_status="NEEDS_CLARIFICATION",
                rtl_code=None,
                testbench_code=None,
                hardware_spec=spec,
                final_report={
                    "status": "NEEDS_CLARIFICATION",
                    "message": (
                        "The requirement is missing information "
                        "needed to generate reliable HDL."
                    ),
                    "questions": spec.get(
                        "clarification_questions",
                        [],
                    ),
                },
            )

        rtl = self.generate_rtl(spec)

        tb = self.generate_testbench(
            spec,
            rtl,
        )

        history = []

        for attempt in range(
            MAX_REPAIR_ATTEMPTS + 1
        ):

            try:
                ok, stage, log = self.run_tools(
                    rtl,
                    tb,
                )

            except FileNotFoundError:
                return AgentResult(
                    verification_status="GENERATED_NOT_SIMULATED",
                    rtl_code=rtl,
                    testbench_code=tb,
                    hardware_spec=spec,
                    attempts=attempt,
                    final_report={
                        "status": "GENERATED_NOT_SIMULATED",
                        "message": (
                            "RTL and testbench were generated, "
                            "but Icarus Verilog is not installed "
                            "on this runtime."
                        ),
                        "history": history,
                    },
                )

            except subprocess.TimeoutExpired:
                ok = False
                stage = "SIMULATION_TIMEOUT"
                log = (
                    "Simulation exceeded the "
                    "20 second safety limit."
                )

            history.append(
                {
                    "attempt": attempt + 1,
                    "stage": stage,
                    "log": log[-4000:],
                }
            )

            if ok:
                return AgentResult(
                    verification_status="VERIFICATION_PASS",
                    rtl_code=rtl,
                    testbench_code=tb,
                    hardware_spec=spec,
                    attempts=attempt + 1,
                    final_report={
                        "status": "VERIFICATION_PASS",
                        "message": (
                            "RTL compiled and the generated "
                            "testbench passed simulation."
                        ),
                        "history": history,
                    },
                )

            if attempt < MAX_REPAIR_ATTEMPTS:
                rtl, tb = self.repair(
                    spec,
                    rtl,
                    tb,
                    stage,
                    log,
                )

        return AgentResult(
            verification_status="VERIFICATION_FAIL",
            rtl_code=rtl,
            testbench_code=tb,
            hardware_spec=spec,
            attempts=MAX_REPAIR_ATTEMPTS + 1,
            final_report={
                "status": "VERIFICATION_FAIL",
                "message": (
                    "Generation completed but automatic "
                    "repair could not make the design "
                    "pass verification."
                ),
                "history": history,
            },
        )


def run_verilog_agent(
    user_request: str,
) -> Dict[str, Any]:
    """Entry point used by the FastAPI application."""

    result = VerilogAgent().run(
        user_request
    )

    return asdict(result)
