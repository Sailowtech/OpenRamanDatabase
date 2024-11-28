from flask import Flask, request, render_template, send_from_directory, redirect
import os
import sqlite3
from app.utils import *

app = Flask(__name__)
db_file_path = 'app/database/microplastics_reference.db'  # Path to SQLite database

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        sample_id = request.form.get('sample_id')
        if file and sample_id:
            # Calculate similarity
            #sample_peaks, normalized_intensities, wave_numbers = process_and_plot_sample(file, sample_id)
            #results, best_match = calculate_similarity(sample_peaks)
            #score = results[best_match]
            best_match, results, plot_file= process_and_compare_sample(file, sample_id)

            # Ensure the function is called with all required arguments
            #add_sample_to_bank(sample_id, normalized_intensities, wave_numbers, best_match, score)
            return render_template('index.html', results=results, best_match=best_match, score=results[best_match], sample_plot=plot_file)

    return render_template('index.html', results=None, best_match=None, score=None, sample_plot=None)

@app.route('/library', methods=['GET', 'POST'])
def library():
    if request.method == 'POST':
        generate_plots()
    
    plots = os.listdir('app/plots')
    
    # Fetch all comments
    comments = {}
    for material_id in reference_spectra_ids:
        _, _, comment = get_spectrum_data(material_id)
        comments[material_id] = comment

    total_ids = len(reference_spectra_ids)
    return render_template('library.html', plots=plots, total_ids=total_ids, comments=comments)

@app.route('/plots/<filename>')
def plot(filename):
    if 'sample' in filename:
        return send_from_directory('sample_plots', filename)
    return send_from_directory('plots', filename)

@app.route('/sample_plots/<filename>')
def sample_plot_retrieve(filename):
    return send_from_directory('sample_plots', filename)

@app.route('/update_comment', methods=['POST'])
def update_comment():
    material_id = request.form.get('material_id')
    new_comment = request.form.get('comment')
    
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE microplastics SET Comment=? WHERE ID=?", (new_comment, material_id))
    conn.commit()
    conn.close()
    
    return redirect('/library')
@app.route('/add_sample', methods=['POST'])
def add_sample():
    sample_id = request.form.get('sample_id')
    file = request.files.get('file')

    if sample_id and file:
        df = process_uploaded_file(file)
        intensities = df.iloc[:, 0].tolist()
        wave_numbers = df.iloc[:, 1].tolist()
        add_sample_to_bank(sample_id, intensities, wave_numbers)

    return redirect('/')

@app.route('/sample_history', methods=['GET'])
def sample_history():
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT sample_id FROM sample_bank")
    samples = cursor.fetchall()
    conn.close()

    sample_ids = [sample[0] for sample in samples]
    #generate_sample_plots(sample_ids)
    plots = os.listdir('app/sample_plots')

    return render_template('sample_history.html', samples=sample_ids, plots=plots)
@app.route('/delete_sample', methods=['POST'])
def delete_sample():
    sample_id = request.form.get('sample_id')
    
    if sample_id:
        # Delete the sample from the database
        conn = sqlite3.connect(db_file_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sample_bank WHERE sample_id=?", (sample_id,))
        conn.commit()
        conn.close()

        # Delete the corresponding plot file
        plot_file_path = os.path.join('app/sample_plots', f'sample_{sample_id}_with_peaks.png')
        if os.path.exists(plot_file_path):
            os.remove(plot_file_path)

    return redirect('/sample_history')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)