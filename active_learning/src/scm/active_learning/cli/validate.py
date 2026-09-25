import sys
from pathlib import Path

from pydantic import ValidationError
from pydantic_settings import BaseSettings, CliApp, CliPositionalArg, SettingsConfigDict

from scm.active_learning import ActiveLearningLoop


class ValidateCli(BaseSettings):
    """Load and validate an active-learning YAML state without executing its configured loop."""

    model_config = SettingsConfigDict(cli_kebab_case=True, cli_implicit_flags=True)

    state_path: CliPositionalArg[Path]

    def cli_cmd(self) -> None:
        """Validate the supplied YAML state by loading its complete active-learning loop model."""
        if self.state_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("state path must be a YAML file ending in .yaml or .yml")
        try:
            ActiveLearningLoop.load_model(self.state_path)
        except ValidationError as exc:
            raise ValueError(f"invalid YAML state: {exc}") from None
        print(f"Valid YAML state: {self.state_path}")


def _prog_name() -> str:
    """Construct this command display name for direct and top-level CLI invocation."""
    prog = Path(sys.argv[0]).name
    return f"{prog} validate" if prog == "al" else sys.argv[0]


def _run_validate_cli(cli_args=None, prog_name=None):
    """Run declarative CLI parsing with an isolated argv for accurate command help."""
    effective_cli_args = sys.argv[1:] if cli_args is None else cli_args
    original_argv = sys.argv.copy()
    try:
        sys.argv = [(prog_name or _prog_name()), *effective_cli_args]
        return CliApp.run(ValidateCli, cli_args=effective_cli_args)
    finally:
        sys.argv = original_argv


def _format_cli_validation_error(exc: ValidationError) -> str:
    """Format argument-validation errors consistently with sibling active-learning CLI commands."""
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
        return f"error: invalid arguments for `al validate`: {'; '.join(other_errors)}"
    return "error: invalid arguments for `al validate`"


def _format_runtime_error(exc: Exception) -> str:
    """Return a concise user-facing error for a YAML state that cannot be loaded."""
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _print_help(prog_name=None) -> None:
    """Display command help after malformed arguments while preserving the original failure."""
    try:
        _run_validate_cli(["--help"], prog_name=prog_name)
    except SystemExit:
        return


def main(cli_args=None, prog_name=None) -> None:
    """Run YAML-only model validation, showing help when invoked directly without arguments."""
    cli_args = ["--help"] if cli_args is None and len(sys.argv) == 1 else cli_args
    try:
        _run_validate_cli(cli_args, prog_name=prog_name)
    except ValidationError as exc:
        _print_help(prog_name=prog_name)
        raise SystemExit(_format_cli_validation_error(exc)) from None
    except Exception as exc:
        raise SystemExit(_format_runtime_error(exc)) from None


if __name__ == "__main__":
    main()
