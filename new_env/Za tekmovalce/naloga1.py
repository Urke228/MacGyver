import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter
from numpy import trapezoid


CSV_FILES = {
    696: Path("C:/Users/ASUS/Documents/MacGyver/new_env/Za tekmovalce/pulse_696.csv"),
    701: Path("C:/Users/ASUS/Documents/MacGyver/new_env/Za tekmovalce/pulse_701.csv"),
    702: Path("C:/Users/ASUS/Documents/MacGyver/new_env/Za tekmovalce/pulse_702.csv"),
}

REFERENCE_DATA = {
    696: {"peak_power": 148.0, "reactivity": 2.0},
    701: {"peak_power": 103.0, "reactivity": 1.9},
    702: {"peak_power": 139.4, "reactivity": 2.0},
}

def get_pulse_id(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if line.lower().startswith("pulse id"):
                return int(line.split(":")[1].strip())
            

def denoise_signal(signal, auto_offset=True, manual_offset=0.0):
    if auto_offset:
        baseline = np.mean(signal[:100])
    else:
        baseline = manual_offset
    corrected = signal - baseline
    corrected[corrected < 0] = 0
    return corrected


def crop_signal(time, signal, threshold_ratio=0.05):
    threshold = np.max(signal)
    cutoff = threshold * threshold_ratio

    indices = np.where(signal > cutoff)[0]

    if len(indices) == 0:
        return time, signal, (0, len(signal)-1)

    start_idx, end_idx = indices[0], indices[-1]

    time_cropped = time.iloc[start_idx:end_idx + 1].reset_index(drop=True)
    signal_cropped = pd.Series(signal[start_idx:end_idx + 1]).reset_index(drop=True)

    return time_cropped, signal_cropped, (start_idx, end_idx)
 

def read_csv_with_metadata(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Poiščemo vrstico, kjer se začnejo podatki (glava Time,Voltage)
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("time"):
            data_start = i
            break

    df = pd.read_csv(file_path, skiprows=data_start)
    return df

def calibrate_signal(signal, peak_power):
    max_val = np.max(signal)
    scale_factor = peak_power / max_val
    return signal * scale_factor, scale_factor

def plot_signals(time, raw_voltage, calibrated_power):
    # === Plot 1: Raw voltage ===
    plt.figure()
    plt.plot(time, raw_voltage, label="Napetost (surovo)", linestyle="--")
    plt.xlabel("Čas [ms]")
    plt.ylabel("Napetost")
    plt.title("Surova napetost")
    plt.legend()
    plt.grid(True)

    # === Plot 2: Calibrated power ===
    plt.figure()
    plt.plot(time, calibrated_power, label="Moč (kalibrirano, MW)")
    plt.xlabel("Čas [ms]")
    plt.ylabel("Moč [MW]")
    plt.title("Kalibrirana moč")
    plt.legend()
    plt.grid(True)

    # === Plot 3: Both on same time axis (for visual alignment) ===
    plt.figure()
    plt.plot(time, raw_voltage, label="Napetost (surovo)", linestyle="--")
    plt.plot(time, calibrated_power, label="Moč (kalibrirano, MW)")
    plt.xlabel("Čas [ms]")
    plt.ylabel("Napetost / Moč")
    plt.title("Primerjava: Napetost in moč")
    plt.legend()
    plt.grid(True)

    plt.show()

def plot_filtered_signals(time, raw_voltage, denoised_voltage, calibrated_power):
    plt.figure(figsize=(10, 6))

    plt.plot(time, raw_voltage, label="Napetost (surova)", linestyle="--", alpha=0.6)
    plt.plot(time, denoised_voltage, label="Napetost (glajena)", linewidth=1.5)
    plt.plot(time, calibrated_power, label="Moč (MW)", linewidth=2)

    plt.xlabel("Čas [ms]")
    plt.ylabel("Napetost / Moč [MW]")
    plt.title("Signal po filtriranju in obrezovanju")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()





def compute_fwhm(time, signal):
    max_val = np.max(signal)
    half_max = max_val / 2

    # Poiščemo indekse, kjer signal prečka polovico maksimuma
    indices = np.where(signal >= half_max)[0]

    if len(indices) < 2:
        print("FWHM ni mogoče izračunati — signal ne doseže polovice maksimuma.")
        return None

    start_idx, end_idx = indices[0], indices[-1]
    fwhm = time[end_idx] - time[start_idx]  # časovna razlika

    return fwhm



def compute_released_energy(time, signal):
    # Pretvori čas iz ms v sekunde za pravilne enote [MW·s]
    time_sec = time / 1000.0
    energy = trapezoid(signal, x=time_sec)
    return energy




def plot_reactivity(time, signal, pulse_id):
    if pulse_id not in REFERENCE_DATA:
        raise ValueError(f"Ni podatkov o $ za Pulse ID {pulse_id}")

    reactivity = REFERENCE_DATA[pulse_id]["reactivity"]

    # Kalibriraj signal z reaktivnostjo namesto z močjo
    max_val = np.max(signal)
    scale_factor = reactivity / max_val
    calibrated_reactivity = signal * scale_factor

    # Obreži kot prej
    time_cropped, reactivity_cropped, _ = crop_signal(time, calibrated_reactivity)

    # === Izris
    plt.figure(figsize=(10, 5))
    plt.plot(time_cropped, reactivity_cropped, label="Reaktivnost $", color="darkorange")
    plt.xlabel("Čas [ms]")
    plt.ylabel("Promptna reaktivnost $")
    plt.title("Graf promptne reaktivnosti skozi čas")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



def process_pulse(pulse_id, file_path):
    print(f"\n--- Obdelava pulza {pulse_id} ---")

    df = read_csv_with_metadata(file_path)
    df.columns = df.columns.str.strip().str.lower()

    time = df["time"].reset_index(drop=True) * 1000  # s → ms
    voltage = df["voltage"]
    
    # Glajenje signala
    voltage_denoised = denoise_signal(voltage)

    # Kalibracija
    peak_power = REFERENCE_DATA[pulse_id]["peak_power"]
    calibrated_power, scale_factor = calibrate_signal(voltage_denoised, peak_power)

    # Obrezovanje
    time_cropped, power_cropped, (start_idx, end_idx) = crop_signal(time, calibrated_power)
    voltage_denoised_cropped = voltage_denoised[start_idx:end_idx + 1]

    # FWHM in energija
    fwhm = compute_fwhm(time_cropped, power_cropped)
    energy = compute_released_energy(time_cropped, power_cropped)

    # Izris grafa moči
    plot_filtered_signals(time_cropped, voltage[start_idx:end_idx + 1], voltage_denoised_cropped, power_cropped)

    # Graf $ (promptne reaktivnosti)
    plot_reactivity(time, voltage_denoised, pulse_id)

    return {
        "pulse_id": pulse_id,
        "peak_power": peak_power,
        "reactivity": REFERENCE_DATA[pulse_id]["reactivity"],
        "scale_factor": scale_factor,
        "fwhm": fwhm,
        "energy": energy
    }


def generate_combined_report(results, output_path="porocilo_vsi_pulzi.txt"):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=== POROČILO ZA VSE PULZE ===\n\n")
        for r in results:
            f.write(f"Pulse ID: {r['pulse_id']}\n")
            f.write(f"  Peak Power: {r['peak_power']} MW\n")
            f.write(f"  Reaktivnost ($): {r['reactivity']}\n")
            f.write(f"  Kalibracijski faktor: {r['scale_factor']:.5f} MW/V\n")
            f.write(f"  FWHM: {r['fwhm']:.3f} ms\n")
            f.write(f"  Released Energy: {r['energy']:.6f} MW·s\n")
            f.write("-" * 40 + "\n")
    print(f"\nSkupno poročilo shranjeno kot: {output_path}")


def main():
    results = []

    for pulse_id, path in CSV_FILES.items():
        result = process_pulse(pulse_id, path)
        results.append(result)

    # Generiraj skupno poročilo
    generate_combined_report(results, output_path="porocilo_vsi_pulzi.txt")



if __name__ == "__main__":
    main()



    
