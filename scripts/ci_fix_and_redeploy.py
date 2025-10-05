"""CI helper to attempt common frontend build fixes and re-run build

This script is intended to be run locally or in CI after a failed frontend build.
It implements a few safe heuristics:
- Ensure `frontend/package.json` has build/start scripts
- Ensure `vite` is a devDependency
- Ensure `homepage` or base path is set for GH Pages
- Attempts to run `npm ci` and `npm run build` and reports results

This is intentionally conservative; it won't rewrite application code.
"""

import json
import subprocess
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / 'frontend'
PACKAGE_JSON = FRONTEND / 'package.json'


def run(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)} (cwd={cwd})")
    result = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout)
    return result.returncode


def ensure_package_json():
    if not PACKAGE_JSON.exists():
        print('frontend/package.json not found.')
        return False
    data = json.loads(PACKAGE_JSON.read_text())
    changed = False
    scripts = data.get('scripts', {})
    if 'build' not in scripts:
        scripts['build'] = 'vite build'
        changed = True
    if 'start' not in scripts:
        scripts['start'] = 'vite preview'
        changed = True
    data['scripts'] = scripts
    if changed:
        PACKAGE_JSON.write_text(json.dumps(data, indent=2))
        print('Updated frontend/package.json with missing scripts')
    return True


def ensure_dev_deps():
    data = json.loads(PACKAGE_JSON.read_text())
    dev = data.get('devDependencies', {})
    if 'vite' not in dev:
        dev['vite'] = '^5.0.0'
        data['devDependencies'] = dev
        PACKAGE_JSON.write_text(json.dumps(data, indent=2))
        print('Added vite to devDependencies')
        return True
    return False


def ensure_homepage():
    data = json.loads(PACKAGE_JSON.read_text())
    if 'homepage' not in data:
        data['homepage'] = f"https://{os.getenv('GITHUB_REPOSITORY', 'Shuvam-Das/AlgoTradingWithZerodha').split('/',1)[0]}.github.io/{os.getenv('GITHUB_REPOSITORY', 'AlgoTradingWithZerodha')}"
        PACKAGE_JSON.write_text(json.dumps(data, indent=2))
        print('Added homepage to package.json')
        return True
    return False


def main():
    print('Starting CI fix helper')
    if not ensure_package_json():
        print('package.json missing, aborting')
        sys.exit(1)
    changed = ensure_dev_deps()
    changed = ensure_homepage() or changed

    # Install deps and attempt build
    rc = run(['npm', 'ci'], cwd=str(FRONTEND))
    if rc != 0:
        print('npm ci failed')
        sys.exit(rc)
    rc = run(['npm', 'run', 'build'], cwd=str(FRONTEND))
    if rc != 0:
        print('Build failed after fixes')
        sys.exit(rc)
    print('Build succeeded')

if __name__ == '__main__':
    main()
