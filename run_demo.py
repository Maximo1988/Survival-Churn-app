import subprocess
import sys
import os
import shutil

project_dir = os.path.dirname(__file__)
outdir = os.path.join(project_dir, 'demos', 'plots')

# Limpiar carpeta de salida antigua
if os.path.exists(outdir):
    for f in os.listdir(outdir):
        fp = os.path.join(outdir, f)
        try:
            os.remove(fp)
        except Exception:
            pass
else:
    os.makedirs(outdir, exist_ok=True)

cmd = [sys.executable, os.path.join(project_dir, 'models.py'), '--save-plots', '--outdir', outdir]
print('Ejecutando:', ' '.join(cmd))
proc = subprocess.run(cmd, capture_output=True, text=True)
print(proc.stdout)
if proc.returncode != 0:
    print('ERROR al ejecutar models.py')
    print(proc.stderr)
    sys.exit(proc.returncode)

saved = os.listdir(outdir)
print(f"Archivos guardados ({len(saved)}):")
for f in saved:
    print(' -', f)

print('Demo completada con éxito.')
