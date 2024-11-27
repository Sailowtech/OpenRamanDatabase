import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
import sqlite3
import numpy as np
import pandas as pd

height_threshold = 0.5

db_file_path = 'app/database/microplastics_reference.db'  # Path to SQLite database

def get_all_ids():
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ID FROM microplastics")
    rows = cursor.fetchall()
    conn.close()
    
    # Extract IDs from the fetched rows
    all_ids = [row[0] for row in rows]
    return all_ids

# Set reference_spectra_ids with all IDs from the database
reference_spectra_ids = get_all_ids()

def get_spectrum_data(material_id):
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    cursor.execute("SELECT Intensity, WaveNumber, Comment FROM microplastics WHERE ID=?", (material_id,))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return [], [], ''

    # Extract intensities and wave numbers from the query result
    intensities, wave_numbers = zip(*[(row[0], row[1]) for row in rows])
    
    # Extract comments, default to empty string if none present
    comments = [row[2] if len(row) > 2 else '' for row in rows]
    # Assuming the comment is the same for all entries of a material_id, take the first one
    comment = comments[0] if comments else ''
    return list(intensities), list(wave_numbers), comment

def process_uploaded_file(file):
    df = pd.read_csv(file)
    return df

def process_spectrum(df):
    intensities = df.iloc[:, 0].tolist()
    wavelengths = df.iloc[:, 1].tolist()
    
    # Normalize intensities
    max_intensity = max(intensities)
    intensities = [i / max_intensity for i in intensities]
        
    peaks, _ = find_peaks(intensities, height=height_threshold)
    peak_wavelengths = [wavelengths[i] for i in peaks]
    peak_intensities = [intensities[i] for i in peaks]
    
    return list(zip(peak_wavelengths, peak_intensities))

def plot_spectrum(wavelengths, intensities, peaks, title, filename, directory='app/plots'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    peak_wavelengths, peak_intensities = zip(*peaks) if peaks else ([], [])
    
    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, intensities, label=title)
    plt.scatter(peak_wavelengths, peak_intensities, color='red', label='Peaks')
    plt.axhline(y=height_threshold, color='green', linestyle='--', label='Threshold')
    plt.xlabel('Wavenumber [/cm]')
    plt.ylabel('Intensity')
    plt.title(f'{title} Spectrum with Peaks')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{directory}/{filename}')
    plt.close()

def calculate_similarity(sample_peaks):
    similarities = {}
    for name in reference_spectra_ids:
        intensities, wavelengths, comment = get_spectrum_data(name)
        
        if not intensities or not wavelengths:
            continue
        
        # Normalize intensities
        max_intensity = max(intensities)
        intensities = [i / max_intensity for i in intensities]
        
        ref_peaks = process_spectrum(pd.DataFrame({'intensity': intensities, 'wavelength': wavelengths}))
        
        # Calculate similarity
        similarity_scores = []
        for sample_peak in sample_peaks:
            for ref_peak in ref_peaks:
                position_diff = abs(sample_peak[0] - ref_peak[0]) / ref_peak[0]
                intensity_diff = abs(sample_peak[1] - ref_peak[1]) / ref_peak[1]
                
                similarity = 1 - (0.5 * position_diff + 0.5 * intensity_diff)
                similarity_scores.append(similarity)
        
        # Average similarity score
        if similarity_scores:
            similarities[name] = np.mean(similarity_scores)
        else:
            similarities[name] = 0
    
    best_match = max(similarities, key=similarities.get) if similarities else None
    return similarities, best_match

def generate_plots():
    for material_id in reference_spectra_ids:
        intensities, wavelengths, comment = get_spectrum_data(material_id)
        
        if not intensities or not wavelengths:
            print(f"No data found for {material_id}")
            continue
        
        peaks, _ = find_peaks(intensities, height=height_threshold)
        peak_wavelengths = [wavelengths[i] for i in peaks]
        peak_intensities = [intensities[i] for i in peaks]
        
        plot_spectrum(
            wavelengths,
            intensities,
            list(zip(peak_wavelengths, peak_intensities)),
            material_id,
            f'{material_id}_with_peaks.png'
        )
    print("All plots generated and saved.")

def process_and_plot_sample(file):
    df = process_uploaded_file(file)
    sample_peaks = process_spectrum(df)
    wavelengths = df.iloc[:, 1].tolist()
    intensities = df.iloc[:, 0].tolist()
    
    max_intensity = max(intensities)
    intensities = [i / max_intensity for i in intensities]
    
    peaks, _ = find_peaks(intensities, height=height_threshold)
    peak_wavelengths = [wavelengths[i] for i in peaks]
    peak_intensities = [intensities[i] for i in peaks]
    
    plot_spectrum(wavelengths, intensities, list(zip(peak_wavelengths, peak_intensities)), 'Sample', 'sample_with_peaks.png', directory='app/sample_plots')
    
    return sample_peaks