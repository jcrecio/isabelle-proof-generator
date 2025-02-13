from datetime import datetime
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time
from timeit import Timer
from typing import Any, List, Optional, TextIO

thy_folder = "/home/jcrecio/repos/isabelle_server/isabelle-proof-generator/afp-current-extractions/thys/Abortable_Linearizable_Modules"

ISABELLE_PATH = "/home/jcrecio/repos/isabelle_server/Isabelle2024/bin/isabelle"
# ISABELLE_COMMAND = "isabelle build -D"
ISABELLE_COMMAND = f"{ISABELLE_PATH} build -D"


def log(
    *values: Any,
    sep: str = " ",
    end: str = "\n",
    file: Optional[TextIO] = None,
    flush: bool = False,
    with_time: bool = True,
) -> None:

    message = sep.join(str(value) for value in values)

    timestamp = f"[{datetime.now()}]" if with_time else ""
    formatted_message = f"{timestamp} {message}"

    output_file = file if file is not None else sys.stdout

    print(formatted_message, end=end, file=output_file, flush=flush)


def convert_to_command(command: str):
    return command.split()


def convert_to_shell_command(command: str):
    return [command]


def run_command_with_output(command, execution_dir=None, timeout=None):
    full_output = []
    full_error = []

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=execution_dir,
        env=os.environ.copy(),
        shell=True,
        preexec_fn=None if timeout is None else os.setsid,
    )

    def kill_process():
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            time.sleep(1)
            if process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    timer = None
    if timeout:
        timer = Timer(timeout, kill_process)
        timer.start()

    try:
        while True:
            output = process.stdout.readline()
            error = process.stderr.readline()

            if not output and not error and process.poll() is not None:
                break

            if output:
                log(output.rstrip())
                full_output.append(output)
                sys.stdout.flush()

            if error:
                log(error.rstrip(), file=sys.stderr)
                full_error.append(error)
                sys.stderr.flush()

        return_code = process.poll()

        if timer and timer.is_alive():
            timer.cancel()

        if return_code is None:
            raise TimeoutError(f"Command execution timed out after {timeout} seconds")

        return return_code, "".join(full_output), "".join(full_error)

    finally:
        if timer:
            timer.cancel()
        if process.poll() is None:
            kill_process()


def read_root_file(file_path: str) -> Optional[str]:
    try:
        with open(file_path, "r") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def extract_theories_files(content: str) -> List[str]:
    theories_pattern = r"theories\s*(.*?)(?=\s+document_files|\s*$)"
    theories_match = re.search(theories_pattern, content, re.DOTALL)

    if not theories_match:
        return []

    theories_text = theories_match.group(1)
    theories = [theory.strip() for theory in theories_text.split() if theory.strip()]

    theory_files = [f"{theory}.thy" for theory in theories]
    return theory_files


def process_root_file(file_path: str) -> List[str]:
    content = read_root_file(file_path)
    if content is None:
        return []

    return extract_theories_files(content)


def check_isabelle_project(project_folder: str) -> bool:
    command_string = f"{ISABELLE_COMMAND} {thy_folder}"
    command = convert_to_shell_command(command_string)
    output = run_command_with_output(command)
    if "error" in output[1]:
        return ["error", output[1]]
    return ["success", output[1]]


if __name__ == "__main__":

    root_path = f"{thy_folder}/ROOT"
    theories = process_root_file(file_path)
