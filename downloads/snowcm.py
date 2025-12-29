import os
import subprocess
from ecmwf.opendata import Client

# --- Laufzeit-Infos aus Umgebungsvariablen lesen ---
date = os.getenv("DATE")        # z. B. "20251105"
time = int(os.getenv("RUN", 0)) # z. B. 0, 12, etc.

# Datum in ISO-Format (YYYY-MM-DD) umwandeln
date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"

# --- Zielordner je nach Variable ---
output_dir = "data/snowcm"  # in msl.py oder ptype.py anpassen
os.makedirs(output_dir, exist_ok=True)

# ECMWF Open Data Client (AWS Mirror empfohlen)
client = Client(source="aws")

# Step-Bereiche gemäß ECMWF-HRES-Definition
steps = list(range(0, 145, 3)) + list(range(150, 361, 6))

# --- 1️⃣ Gesamten Download in eine Datei ---
bigfile = os.path.join(output_dir, "snowcm_all.grib2")
print(f"⬇️ Lade alle Steps auf einmal nach {bigfile}...")

try:
    client.retrieve(
        date=date,
        time=time,
        type="pf",
        step=steps,
        param="sd",   # im jeweiligen Script anpassen
        target=bigfile
    )
except Exception as e:
    print(f"⚠️ Fehler beim Download: {e}")
    exit(1)

# --- 2️⃣ Splitten nach Step ---
print("✂️ Splitte große GRIB in einzelne Step-Dateien...")
try:
    subprocess.run([
        "grib_copy",
        "-w", "stepRange=*",
        bigfile,
        os.path.join(output_dir, "snowcm_step_[stepRange].grib2")
    ], check=True)
except Exception as e:
    print(f"⚠️ Fehler beim Split: {e}")
    exit(1)

# --- 3️⃣ Aufräumen der großen Datei ---
os.remove(bigfile)

print("✅ Alle Daten erfolgreich gespeichert!")
