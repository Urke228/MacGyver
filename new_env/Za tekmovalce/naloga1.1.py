import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pandas as pd

def read_custom_csv(file_path):
    meta = {}
    data_lines = []
    with open(file_path, 'C:/Users/ASUS/Documents/MacGyver/new_env/Za tekmovalce/pulse_696.csv') as file:
        lines = file.readlines()
        parsing_data = False
        for line in lines:
            line = line.strip()
            if not parsing_data:
                if line.startswith('---'):
                    parsing_data = True
                elif ':' in line:
                    key, value = line.split(':', 1)
                    meta[key.strip()] = value.strip()
            else:
                data_lines.append(line)
    
    # Pretvori preostale vrstice v DataFrame
    from io import StringIO
    data_str = '\n'.join(data_lines)
    df = pd.read_csv(StringIO(data_str))
    
    return meta, df

def calibrate_signal(df, peak_power):
    voltage = df['Voltage'].values
    smoothed_voltage = gaussian_filter1d(voltage, sigma=5)
    
    max_voltage = np.max(smoothed_voltage)
    scale_factor = peak_power / max_voltage
    calibrated = smoothed_voltage * scale_factor
    
    df['Calibrated'] = calibrated
    return df, scale_factor

def plot_signal(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df['Time'], df['Calibrated'], label='Kalibriran signal (MW)')
    plt.xlabel('Čas [s]')
    plt.ylabel('Moč [MW]')
    plt.title('Kalibriran signal pulza')
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    meta, df = read_custom_csv("data/pulz1.csv")
    df, factor = calibrate_signal(df, peak_power=148.0)
    plot_signal(df)
    print(f"Skalirni faktor: {factor:.3f} (MW/V)")


if __name__ == "__main__":
    main()
