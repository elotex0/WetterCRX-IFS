import os
from ecmwf.opendata import Client

# Zielordner definieren
output_dir = "data/t2m"
os.makedirs(output_dir, exist_ok=True)  # erstellt Ordner, falls nicht vorhanden

# ECMWF Open Data Client (AWS Mirror empfohlen)
client = Client(source="aws")

# Step-Bereiche gemäß ECMWF-HRES-Definition
steps = list(range(0, 145, 3)) + list(range(150, 361, 6))

# Datum & Lauf (Beispiel: 00 UTC Lauf vom 2025-11-05)
date = "2025-11-05"
time = 0

for step in steps:
    filename = f"2t_step_{step:03d}.grib2"
    target_path = os.path.join(output_dir, filename)

    print(f"Lade Step +{step:03d}h herunter → {target_path}")
    try:
        client.retrieve(
            date=date,
            time=time,
            type="fc",
            step=step,
            param="2t",
            target=target_path
        )
    except Exception as e:
        print(f"⚠️ Fehler bei Step {step}: {e}")

print("✅ Alle 2t-Daten erfolgreich in data/t2m gespeichert!")
