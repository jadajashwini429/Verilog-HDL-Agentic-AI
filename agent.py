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


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")

MAX_REPAIR_ATTEMPTS = int(
    os.getenv("MAX_REPAIR_ATTEMPTS", "2")
)

IVERILOG = os.getenv(
    "IVERILOG_PATH",
    "iverilog"
)

VVP = os.getenv(
    "VVP_PATH",
    "vvp"
)

LLM_RETRIES = int(
    os.getenv("LLM_RETRIES", "4")
)


# ============================================================
# RESULT STRUCTURE
# ============================================================

@dataclass
class AgentResult:
    verification_status: str
    rtl_code: str | None
    testbench_code: str | None
    final_report: Dict[str, Any]
    hardware_spec: Dict[str, Any] | None = None
    attempts: int = 0


# ============================================================
# VERILOG AGENT
# ============================================================

class VerilogAgent:
    """
    Multi-stage Verilog generation and verification agent.

    Pipeline:

    1. Requirement understanding
    2. RTL generation
    3. Testbench generation
    4. Compilation
    5. Simulation
    6. Automatic repair
    7. Final report
    """

    def __init__(self, api_key: str | None = None):

        key = api_key or os.getenv(
            "GEMINI_API_KEY"
        )

        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured."
            )

        if genai is None:
            raise RuntimeError(
                "google-genai is not installed."
            )

        self.client = genai.Client(
            api_key=key
        )

    # ========================================================
    # LLM CALL
    # ========================================================

    def llm(
        self,
        system: str,
        user: str,
    ) -> str:

        prompt = (
            f"SYSTEM ROLE:\n"
            f"{system}\n\n"
            f"USER INPUT:\n"
            f"{user}\n\n"
            "Return only the requested output. "
            "Do not add explanations unless explicitly requested."
        )

        last_error = None

        for attempt in range(LLM_RETRIES):

            try:

                print(
                    f"[LLM] Requesting {MODEL} "
                    f"(attempt {attempt + 1}/{LLM_RETRIES})"
                )

                # IMPORTANT:
                # Gemini 3.7 does NOT use the old
                # temperature/top_p/top_k parameters.
                response = self.client.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                )

                text = getattr(
                    response,
                    "text",
                    None
                )

                if not text:
                    raise RuntimeError(
                        "The model returned an empty response."
                    )

                print(
                    "[LLM] Response received successfully."
                )

                return text.strip()

            except Exception as exc:

                last_error = exc

                error_text = str(exc).lower()

                print(
                    f"[LLM ERROR] {exc}"
                )

                temporary_error = any(
                    keyword in error_text
                    for keyword in [
                        "503",
                        "unavailable",
                        "429",
                        "resource exhausted",
                        "high demand",
                        "temporarily",
                        "deadline",
                        "timeout",
                    ]
                )

                if not temporary_error:
                    raise

                if attempt == LLM_RETRIES - 1:
                    break

                # Exponential backoff:
                # 2s, 4s, 8s...
                wait_time = min(
                    2 ** attempt,
                    15
                )

                print(
                    f"[LLM] Temporary failure. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

        raise RuntimeError(
            f"LLM request failed after "
            f"{LLM_RETRIES} attempts: {last_error}"
        )

    # ========================================================
    # CLEAN GENERATED CODE
    # ========================================================

    @staticmethod
    def clean_code(
        text: str,
        language: str = "verilog"
    ) -> str:

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

    # ========================================================
    # JSON PARSER
    # ========================================================

    @staticmethod
    def parse_json(
        text: str
    ) -> Dict[str, Any]:

        text = text.strip()

        if text.startswith("```"):

            text = re.sub(
                r"^```(?:json)?\s*",
                "",
                text,
                flags=re.IGNORECASE,
            )

            text = re.sub(
                r"\s*```$",
                "",
                text,
            )

        try:

            return json.loads(text)

        except json.JSONDecodeError:

            match = re.search(
                r"\{.*\}",
                text,
                re.DOTALL,
            )

            if match:

                return json.loads(
                    match.group(0)
                )

            raise

    # ========================================================
    # REQUIREMENT UNDERSTANDING
    # ========================================================

    def understand(
        self,
        request: str
    ) -> Dict[str, Any]:

        system = """
You are the Requirement Understanding Agent
for a Verilog HDL automation system.

Convert natural-language hardware requirements
into a precise implementable specification.

If critical information is missing:

- Do NOT invent it.
- Set clarification_needed=true.
- Provide concise clarification questions.

If enough information exists:

- Infer only conventional details.
- Set clarification_needed=false.

Target Verilog-2001/Verilog HDL,
not SystemVerilog.

Return JSON only with exactly these fields:

{
  "clarification_needed": boolean,
  "clarification_questions": [string],
  "module_name": string,
  "description": string,
  "parameters": [
    {
      "name": string,
      "value": string
    }
  ],
  "inputs": [
    {
      "name": string,
      "width": integer,
      "description": string
    }
  ],
  "outputs": [
    {
      "name": string,
      "width": integer,
      "description": string
    }
  ],
  "behavior": [string],
  "clocking": string,
  "reset": string,
  "assumptions": [string]
}
"""

        spec = self.parse_json(
            self.llm(
                system,
                request,
            )
        )

        spec.setdefault(
            "parameters",
            []
        )

        spec.setdefault(
            "inputs",
            []
        )

        spec.setdefault(
            "outputs",
            []
        )

        spec.setdefault(
            "behavior",
            []
        )

        spec.setdefault(
            "assumptions",
            []
        )

        spec.setdefault(
            "clarification_questions",
            []
        )

        return spec

    # ========================================================
    # RTL GENERATION
    # ========================================================

    def generate_rtl(
        self,
        spec: Dict[str, Any]
    ) -> str:

        system = """
You are the RTL Generation Agent.

Generate synthesizable Verilog HDL only.

Rules:

- Use Verilog-2001.
- Do NOT use SystemVerilog.
- Generate exactly one top-level module.
- Use the specified module_name.
- Match every port name exactly.
- Match every port width exactly.
- Do not generate a testbench.
- Do not generate explanations.
- Do not use Markdown.
- Prefer simple synthesizable constructs.
- For combinational logic use assign or always @*.
- For sequential logic use always @(posedge clock).
- Follow the specified reset behavior.
- Do not create undeclared ports.
- Do not create undeclared signals.
- Finish with endmodule.
"""

        return self.clean_code(
            self.llm(
                system,
                json.dumps(
                    spec,
                    indent=2
                ),
            )
        )

    # ========================================================
    # TESTBENCH GENERATION
    # ========================================================

    def generate_testbench(
        self,
        spec: Dict[str, Any],
        rtl: str,
    ) -> str:

        system = """
You are the Verification/Testbench Agent
for Verilog HDL.

Generate a self-contained Verilog-2001
testbench for the supplied DUT.

Rules:

- Testbench module must be named:
  tb_<dut_module_name>

- Instantiate the DUT using exactly
  its declared ports.

- Exercise representative normal cases.

- Exercise important boundary/corner cases.

- For combinational circuits:
  use deterministic checks with if statements.

- For sequential circuits:
  generate the required clock/reset.

- Allow enough simulation time.

- Print exactly:

VERIFICATION_PASS

when all checks pass.

- Print:

VERIFICATION_FAIL

when any check fails.

- End with $finish.

- Do NOT use SystemVerilog-only features.

- Return code only.
"""

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
            )
        )

    # ========================================================
    # COMPILE + SIMULATE
    # ========================================================

    def run_tools(
        self,
        rtl: str,
        tb: str,
    ) -> Tuple[bool, str, str]:

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

            # ------------------------------------------------
            # COMPILE
            # ------------------------------------------------

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

            # ------------------------------------------------
            # SIMULATE
            # ------------------------------------------------

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

    # ========================================================
    # REPAIR
    # ========================================================

    def repair(
        self,
        spec: Dict[str, Any],
        rtl: str,
        tb: str,
        stage: str,
        log: str,
    ) -> Tuple[str, str]:

        system = """
You are the Debug and Repair Agent
for a Verilog generation pipeline.

A generated design/testbench failed
compilation or simulation.

Analyze the failure and return JSON only:

{
  "rtl_code": "...",
  "testbench_code": "..."
}

Rules:

- Preserve the user's intended behavior.
- Fix only what is necessary.
- Keep Verilog-2001 compatibility.
- Ensure DUT/testbench interfaces agree exactly.
- Ensure the testbench prints VERIFICATION_PASS
  on success.
- Ensure the testbench prints VERIFICATION_FAIL
  on failure.
- Never put Markdown fences around code
  inside JSON strings.
"""

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
            self.clean_code(
                repaired_rtl
            ),
            self.clean_code(
                repaired_tb
            ),
        )

    # ========================================================
    # COMPLETE PIPELINE
    # ========================================================

    def run(
        self,
        request: str
    ) -> AgentResult:

        print(
            "[AGENT] Understanding requirement..."
        )

        spec = self.understand(
            request
        )

        # ----------------------------------------------------
        # CLARIFICATION
        # ----------------------------------------------------

        if spec.get(
            "clarification_needed"
        ):

            return AgentResult(
                verification_status=(
                    "NEEDS_CLARIFICATION"
                ),
                rtl_code=None,
                testbench_code=None,
                hardware_spec=spec,
                final_report={
                    "status": (
                        "NEEDS_CLARIFICATION"
                    ),
                    "message": (
                        "The requirement is missing "
                        "information needed to generate "
                        "reliable HDL."
                    ),
                    "questions": spec.get(
                        "clarification_questions",
                        [],
                    ),
                },
            )

        # ----------------------------------------------------
        # RTL
        # ----------------------------------------------------

        print(
            "[AGENT] Generating RTL..."
        )

        rtl = self.generate_rtl(
            spec
        )

        # ----------------------------------------------------
        # TESTBENCH
        # ----------------------------------------------------

        print(
            "[AGENT] Generating testbench..."
        )

        tb = self.generate_testbench(
            spec,
            rtl,
        )

        history = []

        # ----------------------------------------------------
        # VERIFY + REPAIR LOOP
        # ----------------------------------------------------

        for attempt in range(
            MAX_REPAIR_ATTEMPTS + 1
        ):

            print(
                f"[AGENT] Verification attempt "
                f"{attempt + 1}/"
                f"{MAX_REPAIR_ATTEMPTS + 1}"
            )

            try:

                ok, stage, log = self.run_tools(
                    rtl,
                    tb,
                )

            except FileNotFoundError:

                return AgentResult(
                    verification_status=(
                        "GENERATED_NOT_SIMULATED"
                    ),
                    rtl_code=rtl,
                    testbench_code=tb,
                    hardware_spec=spec,
                    attempts=attempt,
                    final_report={
                        "status": (
                            "GENERATED_NOT_SIMULATED"
                        ),
                        "message": (
                            "RTL and testbench were "
                            "generated, but Icarus "
                            "Verilog is not installed "
                            "on this runtime."
                        ),
                        "history": history,
                    },
                )

            except subprocess.TimeoutExpired:

                ok = False

                stage = (
                    "SIMULATION_TIMEOUT"
                )

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

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            if ok:

                print(
                    "[AGENT] VERIFICATION_PASS"
                )

                return AgentResult(
                    verification_status=(
                        "VERIFICATION_PASS"
                    ),
                    rtl_code=rtl,
                    testbench_code=tb,
                    hardware_spec=spec,
                    attempts=attempt + 1,
                    final_report={
                        "status": (
                            "VERIFICATION_PASS"
                        ),
                        "message": (
                            "RTL compiled and the "
                            "generated testbench "
                            "passed simulation."
                        ),
                        "history": history,
                    },
                )

            # ------------------------------------------------
            # REPAIR
            # ------------------------------------------------

            if attempt < MAX_REPAIR_ATTEMPTS:

                print(
                    "[AGENT] Attempting automatic repair..."
                )

                rtl, tb = self.repair(
                    spec,
                    rtl,
                    tb,
                    stage,
                    log,
                )

        # ----------------------------------------------------
        # FAILURE AFTER REPAIR
        # ----------------------------------------------------

        return AgentResult(
            verification_status=(
                "VERIFICATION_FAIL"
            ),
            rtl_code=rtl,
            testbench_code=tb,
            hardware_spec=spec,
            attempts=(
                MAX_REPAIR_ATTEMPTS + 1
            ),
            final_report={
                "status": (
                    "VERIFICATION_FAIL"
                ),
                "message": (
                    "Generation completed but "
                    "automatic repair could not "
                    "make the design pass "
                    "verification."
                ),
                "history": history,
            },
        )


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def run_verilog_agent(
    user_request: str,
) -> Dict[str, Any]:

    result = VerilogAgent().run(
        user_request
    )

    return asdict(result)
