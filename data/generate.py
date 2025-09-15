import json
import random
import pandas as pd

names = ["Anna Green", "Brian Lee", "Catherine Hall", "Daniel Young", "Emily Stone",
         "Frank Moore", "Grace White", "Henry Black", "Isabel King", "Jack Wright"]

ids = [f"V-{100+i}" for i in range(10)]
dobs = ["1980-01-01", "1975-05-22", "1990-03-14", "1985-09-09", "1978-12-25",
        "1982-05-11", "1973-08-30", "1992-04-19", "1988-10-07", "1981-06-21"]

bps = [(118, 76), (125, 82), (122, 78), (130, 85), (115, 75),
       (128, 80), (135, 88), (110, 70), (120, 79), (124, 81)]

hrs = [72, 80, 76, 85, 70, 78, 82, 68, 74, 79]
spo2s = [96, 94, 97, 95, 98, 93, 92, 99, 97, 95]
temps = [37.2, 98.6, 36.8, 99.1, 37.0, 100.2, 36.7, 97.9, 37.4, 98.2]

def make_record(name, pid, dob, bp, hr, spo2, temp, allow_missing=True):
    fields = {
        "name": name,
        "patient_id": pid,
        "date_of_birth": dob,
        "blood_pressure_systolic": str(bp[0]),
        "blood_pressure_diastolic": str(bp[1]),
        "heart_rate": str(hr),
        "spo2": str(spo2),
        "temperature": str(temp),
    }

    # randomly drop some fields to simulate missing data
    if allow_missing:
        drop_keys = random.sample(list(fields.keys()), k=random.randint(0, 3))
        for k in drop_keys:
            del fields[k]

    # random input styles
    input_parts = [f"Patient Name: {name}", f"Patient ID: {pid}", f"DOB: {dob}"]

    if "blood_pressure_systolic" in fields and "blood_pressure_diastolic" in fields:
        input_parts.append(f"BP: {bp[0]}/{bp[1]} mmHg")
    if "heart_rate" in fields:
        input_parts.append(f"Pulse: {hr} bpm")
    if "spo2" in fields:
        input_parts.append(f"O₂ Sat: {spo2}%")
    if "temperature" in fields:
        unit = "C" if temp < 60 else "F"
        input_parts.append(f"Temp: {temp} {unit}")

    # add some irrelevant info
    if random.random() > 0.7:
        input_parts.append(f"Doctor: Dr. {random.choice(['Smith','Brown','Patel'])}")

    return "\n".join(input_parts), json.dumps(fields)

# ===== Generate validation (10 samples) =====
val_rows = []
for i in range(10):
    inp, tgt = make_record(names[i], ids[i], dobs[i], bps[i], hrs[i], spo2s[i], temps[i])
    val_rows.append({"input_text": inp, "target_text": tgt})
pd.DataFrame(val_rows).to_csv("val.tsv", sep="\t", index=False)

# ===== Generate test (10 samples) =====
test_rows = []
for i in range(10):
    inp, tgt = make_record(names[i], f"T-{200+i}", dobs[i], bps[i], hrs[i], spo2s[i], temps[i])
    test_rows.append({"input_text": inp, "target_text": tgt})
pd.DataFrame(test_rows).to_csv("test.tsv", sep="\t", index=False)

print("✅ Generated val.tsv and test.tsv with random variations")
