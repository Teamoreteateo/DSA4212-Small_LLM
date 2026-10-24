"""Local, offline checks of the environment and repository (no cloud, no downloads).
  python scripts/check_env.py               # all checks
  python scripts/check_env.py --no-installed  # skip comparing installed package versions
Exits 1 if any check fails; warnings do not fail."""
import argparse, ast, json, re, subprocess, sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_TRACKED_BYTES = 5 * 1024 * 1024
FORBIDDEN = re.compile(r"(^|/)(\.env(\..*)?|\.modal\.toml|secrets/.*)$|\.(pem|key|pt|pth|ckpt|safetensors|bin|npy|npz|arrow|parquet)$")
PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9.+!-]+)$")
failures, warnings = [], []

def fail(msg): failures.append(msg); print(f"  FAIL {msg}")
def warn(msg): warnings.append(msg); print(f"  WARN {msg}")
def ok(msg): print(f"  ok   {msg}")

def read_pins(path, seen=None):
    """Return {name: version} from a requirements file, following -r includes."""
    seen = seen if seen is not None else set()
    pins = {}
    for n, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            inc = path.parent / line[3:].strip()
            if inc in seen:
                fail(f"{path.name}:{n} include cycle via {inc.name}"); continue
            seen.add(inc)
            pins.update(read_pins(inc, seen)); continue
        m = PIN.match(line)
        if not m:
            fail(f"{path.name}:{n} '{line}' is not an exact '==' pin"); continue
        name = re.sub(r"[-_.]+", "-", m.group(1)).lower()
        if name in pins:
            fail(f"{path.name}:{n} {name} pinned twice")
        pins[name] = m.group(2)
    return pins

def check_requirements(check_installed):
    print("requirements")
    pins = read_pins(ROOT / "requirements-dev.txt")
    ok(f"{len(pins)} exact pins: " + ", ".join(f"{k}=={v}" for k, v in pins.items()))
    if "modal" in read_pins(ROOT / "requirements.txt"):
        warn("modal is in requirements.txt; it belongs in requirements-dev.txt only")
    if not check_installed:
        return
    for name, want in pins.items():
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            warn(f"{name} not installed (pip install -r requirements-dev.txt)"); continue
        base = have.split("+", 1)[0]  # torch may report e.g. 2.11.0+cpu
        (ok if base == want else fail)(f"{name} installed {have}" + ("" if base == want else f", expected {want}"))

def check_python():
    print("python")
    tree = ast.parse((ROOT / "modal_app.py").read_text())
    image_py = next((n.value.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                     and any(getattr(t, "id", None) == "PYTHON_VERSION" for t in n.targets)), None)
    if image_py is None:
        fail("PYTHON_VERSION not found in modal_app.py"); return
    local = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info < (3, 11):
        fail(f"local Python {local} < 3.11 (numpy==2.3.2 needs >=3.11)")
    elif local != image_py:
        warn(f"local Python {local} differs from Modal image Python {image_py}")
    else:
        ok(f"local Python {local} matches Modal image")

def check_compile():
    print("python files compile")
    files = [ROOT / f for f in tracked() if f.endswith(".py")]
    files += [p for p in (ROOT / "scripts").glob("*.py") if p not in files]
    for p in files:
        try:
            compile(p.read_text(encoding="utf-8"), str(p), "exec")  # syntax only, writes nothing
        except SyntaxError as e:
            fail(f"{p.relative_to(ROOT)}:{e.lineno}: {e.msg}")
    ok(f"{len(files)} files checked")

def check_configs():
    print("configs")
    for p in sorted((ROOT / "configs").glob("*.json")):
        text = p.read_text().strip()
        if not text:
            ok(f"{p.name} is an empty placeholder (skipped)"); continue
        try:
            json.loads(text); ok(f"{p.name} parses")
        except json.JSONDecodeError as e:
            fail(f"{p.name}: {e}")

def tracked():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [f for f in out.stdout.split("\0") if f]

def check_tracked():
    print("tracked files")
    files = tracked()
    for f in files:
        if FORBIDDEN.search(f):
            fail(f"{f} looks like a credential, checkpoint or data file")
        p = ROOT / f
        if p.is_file() and p.stat().st_size > MAX_TRACKED_BYTES:
            fail(f"{f} is larger than {MAX_TRACKED_BYTES // 2**20} MB")
    ok(f"{len(files)} tracked files scanned")

def check_torch():
    print("torch (local)")
    try:
        import torch
    except ImportError:
        warn("torch not importable; skipped"); return
    cuda = torch.cuda.is_available()
    ok(f"torch {torch.__version__}, CUDA build {torch.version.cuda}, GPU available: {cuda}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-installed", action="store_true", help="skip installed-version and torch checks")
    args = ap.parse_args()
    check_python()
    check_requirements(not args.no_installed)
    check_compile()
    check_configs()
    check_tracked()
    if not args.no_installed:
        check_torch()
    print(f"\n{len(failures)} failure(s), {len(warnings)} warning(s)")
    sys.exit(1 if failures else 0)

if __name__ == "__main__":
    main()
