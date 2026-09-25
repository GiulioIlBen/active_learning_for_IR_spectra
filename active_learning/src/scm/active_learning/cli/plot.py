import sys
from pathlib import Path

from pydantic import ValidationError
from pydantic_settings import BaseSettings, CliApp, CliPositionalArg, SettingsConfigDict

from scm.active_learning import ActiveLearningLoop


class PlotCli(BaseSettings):
    model_config = SettingsConfigDict(cli_kebab_case=True, cli_implicit_flags=True)

    state_path: CliPositionalArg[Path]
    name: str = "analysis.pdf"
    folder: str = "<ALFolder>"
    long: bool = False

    def cli_cmd(self) -> None:
        ActiveLearningLoop.load_model(self.state_path).analysis.plot.save_to_pdf(self.name, self.folder)


def _prog_name() -> str:
    prog = Path(sys.argv[0]).name
    return f"{prog} plot" if prog == "al" else sys.argv[0]


def _run_plot_cli(cli_args=None, prog_name=None):
    effective_cli_args = sys.argv[1:] if cli_args is None else cli_args
    original_argv = sys.argv.copy()
    try:
        sys.argv = [(prog_name or _prog_name()), *effective_cli_args]
        return CliApp.run(PlotCli, cli_args=effective_cli_args)
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
        return f"error: invalid arguments for `al plot`: {'; '.join(other_errors)}"
    return "error: invalid arguments for `al plot`"


def _long_error_requested(cli_args=None) -> bool:
    effective_cli_args = sys.argv[1:] if cli_args is None else cli_args
    return "--long" in effective_cli_args


def _format_runtime_error(exc: Exception) -> str:
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _print_help(prog_name=None) -> None:
    try:
        _run_plot_cli(["--help"], prog_name=prog_name)
    except SystemExit:
        return


def main(cli_args=None, prog_name=None) -> None:
    cli_args = ["--help"] if cli_args is None and len(sys.argv) == 1 else cli_args
    try:
        _run_plot_cli(cli_args, prog_name=prog_name)
    except ValidationError as exc:
        _print_help(prog_name=prog_name)
        if _long_error_requested(cli_args):
            raise
        raise SystemExit(_format_cli_validation_error(exc)) from None
    except Exception as exc:
        if _long_error_requested(cli_args):
            raise
        raise SystemExit(_format_runtime_error(exc)) from None


if __name__ == "__main__":
    main()
