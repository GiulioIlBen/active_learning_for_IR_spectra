import sys
from pathlib import Path

from pydantic import ValidationError
from pydantic_settings import BaseSettings, CliApp, CliPositionalArg, SettingsConfigDict
from scm.moliterate.utils.progress_bar import MOLITERATE_PROGRESS_BAR_CONFIG
from scm.plams import config as plams_config

from scm.active_learning import ActiveLearningLoop


class RunCli(BaseSettings):
    model_config = SettingsConfigDict(cli_kebab_case=True, cli_implicit_flags=True)

    model_path: CliPositionalArg[Path]
    resume: bool = False
    moliterate_progress: bool = False
    plams_stdout: int = 0

    def cli_cmd(self) -> None:
        MOLITERATE_PROGRESS_BAR_CONFIG.disable = not self.moliterate_progress
        plams_config.log.stdout = self.plams_stdout
        al_loop = ActiveLearningLoop.load_model(self.model_path)
        if self.resume:
            al_loop.run_control.mode = "resume"
        al_loop.run()


def _prog_name() -> str:
    prog = Path(sys.argv[0]).name
    return f"{prog} run" if prog == "al" else sys.argv[0]


def _run_run_cli(cli_args=None, prog_name=None):
    effective_cli_args = sys.argv[1:] if cli_args is None else cli_args
    original_argv = sys.argv.copy()
    try:
        sys.argv = [(prog_name or _prog_name()), *effective_cli_args]
        return CliApp.run(RunCli, cli_args=effective_cli_args)
    finally:
        sys.argv = original_argv


def _format_cli_validation_error(exc: ValidationError) -> str:
    missing_args = []
    other_errors = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        if error["type"] == "missing" and len(error["loc"]) == 1 and isinstance(error["loc"][0], str):
            missing_args.append(f"--{error['loc'][0].replace('_', '-')}")
            continue
        other_errors.append(f"{location}: {error['msg']}")

    if missing_args:
        suffix = "s" if len(missing_args) > 1 else ""
        return f"error: missing required argument{suffix}: {', '.join(missing_args)}"
    if other_errors:
        return f"error: invalid arguments for `al run`: {'; '.join(other_errors)}"
    return "error: invalid arguments for `al run`"


def _print_help(prog_name=None) -> None:
    try:
        _run_run_cli(["--help"], prog_name=prog_name)
    except SystemExit:
        return


def main(cli_args=None, prog_name=None) -> None:
    cli_args = ["--help"] if cli_args is None and len(sys.argv) == 1 else cli_args
    try:
        _run_run_cli(cli_args, prog_name=prog_name)
    except ValidationError as exc:
        _print_help(prog_name=prog_name)
        raise SystemExit(_format_cli_validation_error(exc)) from None


if __name__ == "__main__":
    main()
