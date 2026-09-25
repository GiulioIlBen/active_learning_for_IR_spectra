import sys

from scm.active_learning.cli.copy import main as copy_main
from scm.active_learning.cli.plot import main as plot_main
from scm.active_learning.cli.run import main as run_main
from scm.active_learning.cli.validate import main as validate_main


def main(cli_args=None) -> None:
    cli_args = sys.argv[1:] if cli_args is None else cli_args
    if not cli_args or cli_args[0] in {"-h", "--help"}:
        print("usage: al <command> [<args>]\n\ncommands:\n  copy\n  plot\n  run\n  validate")
        raise SystemExit(0)
    if cli_args[0] == "copy":
        copy_main(cli_args[1:], prog_name="al copy")
        return
    if cli_args[0] == "plot":
        plot_main(cli_args[1:], prog_name="al plot")
        return
    if cli_args[0] == "run":
        run_main(cli_args[1:], prog_name="al run")
        return
    if cli_args[0] == "validate":
        validate_main(cli_args[1:], prog_name="al validate")
        return
    raise SystemExit(f"unknown command: {cli_args[0]}")


if __name__ == "__main__":
    main()
