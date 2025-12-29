import os
from ecmwf.opendata import Client
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Laufzeit-Infos aus Umgebungsvariablen lesen ---
date_env = os.getenv("DATE")        # z. B. "20251105"
time_env = int(os.getenv("RUN", 0)) # z. B. 0, 12, etc.

# Datum in ISO-Format (YYYY-MM-DD) umwandeln
date = f"{date_env[:4]}-{date_env[4:6]}-{date_env[6:8]}"
time = time_env

# --- Zielordner ---
output_dir = "data/snowcm"
os.makedirs(output_dir, exist_ok=True)

# ECMWF Open Data Client (AWS Mirror empfohlen)
client = Client(source="aws")

# Step-Bereiche gemäß ECMWF-HRES-Definition
steps = list(range(0, 145, 3)) + list(range(150, 361, 6))

# Ensemble-Mitglieder (0 = control, 1-50 = perturbed members)
members = list(range(1, 51))

# Funktion für den einzelnen Download
def download_member(step, member):
    filename = f"snowcm_step_{step:03d}_member{member:02d}.grib2"
    target_path = os.path.join(output_dir, filename)
    try:
        client.retrieve(
            date=date,
            time=time,
            type="pf",
            step=step,
            number=member,  # einzelnes Mitglied
            param="sd",      # anpassen, falls andere Parameter nötig
            target=target_path
        )
        print(f"✅ Download fertig: Step +{step:03d}h, Member {member:02d}")
    except Exception as e:
        print(f"⚠️ Fehler bei Step +{step:03d}h, Member {member:02d}: {e}")

# Anzahl gleichzeitiger Threads
max_workers = 8

# Parallel Download aller Steps und Mitglieder
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(download_member, step, member) for step in steps for member in members]
    for future in as_completed(futures):
        pass  # Statusmeldungen kommen direkt aus download_member()

print(f"✅ Alle Daten erfolgreich in {output_dir} gespeichert!")
