from pathlib import Path

from scm.active_learning import ActiveLearningLoop


def latest_final_state(alruns: Path = Path("ALruns")) -> Path:
    states = [path for path in alruns.glob("*/al_final_state.*") if path.suffix.lower() in {".json", ".yaml", ".yml"}]
    if not states:
        raise FileNotFoundError(f"No final Active Learning state found below {alruns}")
    return max(states, key=lambda path: path.stat().st_mtime)


def main(state_path: str | Path | None = None):
    al = ActiveLearningLoop.load_model(state_path or latest_final_state())
    an = al.analysis
    print(an.view_table("summary"))


if __name__ == "__main__":
    main()
