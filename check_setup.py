#!/usr/bin/env python3
"""Run this first. Tells you whether this Python can run the pipeline, and if not,
exactly what to install."""
import importlib, platform, sys

print("interpreter  %s" % sys.executable)
print("python       %s on %s" % (platform.python_version(), platform.system()))
print()
need = {'numpy': 'numpy', 'cv2': 'opencv-python', 'scipy': 'scipy'}
missing = []
for mod, pkg in need.items():
    try:
        m = importlib.import_module(mod)
        print("  %-8s OK    %s" % (mod, getattr(m, '__version__', '?')))
    except ImportError:
        print("  %-8s MISSING  -> pip install %s" % (mod, pkg))
        missing.append(pkg)

if missing:
    print("\nInstall what's missing:\n    %s -m pip install %s" % (sys.executable, " ".join(missing)))
    sys.exit(1)

try:
    sys.path.insert(0, __file__.rsplit('\\', 1)[0].rsplit('/', 1)[0])
    from lib import tooling, flow, fields, mesh
except Exception as e:
    print("\nlib/ did not import: %s" % e)
    print("Run this from inside the relief_pipeline folder.")
    sys.exit(1)

print("\n  lib/     OK    tooling, flow, fields, mesh")
print("\nSanity check on the tool model:")
print("  1/16in tapered ball, tip R 0.794 mm, taper 5.36 deg")
for d in (2.0, 3.0, 5.0, 7.0):
    r = tooling.effective_radius(d, 0.79375, 5.36)
    print("    at %.0f mm depth -> effective R %.3f mm  (min feature %.2f mm wide)" % (d, r, 2 * r))
print("\nReady. See README.md for the two entry points.")
