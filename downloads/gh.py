import os
from ecmwf.opendata import Client

# --- Laufzeit-Infos aus Umgebungsvariablen lesen ---
date = os.getenv("DATE" "20251113")        # z. B. "20251105"
time = int(os.getenv("RUN", 12)) # z. B. 0, 12, etc.

# Datum in ISO-Format (YYYY-MM-DD) umwandeln
date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"

# --- Zielordner je nach Variable ---
output_dir = "data/geo"  # in msl.py oder ptype.py anpassen
os.makedirs(output_dir, exist_ok=True)

# ECMWF Open Data Client (AWS Mirror empfohlen)
client = Client(source="aws")

# Step-Bereiche gemäß ECMWF-HRES-Definition
steps = list(range(0, 145, 3)) + list(range(150, 361, 6))

for step in steps:
    filename = f"gh_step_{step:03d}.grib2"  # im jeweiligen Script anpassen
    target_path = os.path.join(output_dir, filename)

    print(f"Lade Step +{step:03d}h für {date} {time:02d} UTC → {target_path}")
    try:
        client.retrieve(
            date=date,
            time=time,
            type="fc",
            levelist="500",
            step=step,
            param="gh",   # im jeweiligen Script anpassen
            target=target_path
        )
    except Exception as e:
        print(f"⚠️ Fehler bei Step {step}: {e}")

print("✅ Alle Daten erfolgreich gespeichert!")
