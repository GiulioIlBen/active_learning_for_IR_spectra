import sys

from scm.moliterate.cli.view_cli import main as view_main


def main(cli_args=None) -> int:
    cli_args = sys.argv[1:] if cli_args is None else cli_args
    if not cli_args or cli_args[0] in {"-h", "--help"}:
        print("usage: moliterate <command> [<args>]\n\ncommands:\n  view")
        return 0
    if cli_args[0] == "view":
        return view_main(cli_args[1:])
    raise SystemExit(f"unknown command: {cli_args[0]}")


if __name__ == "__main__":
    raise SystemExit(main())
