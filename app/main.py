from flask import Flask, request, render_template, send_from_directory, redirect
import os
import sqlite3
from app.utils import process_and_plot_sample, calculate_similarity, generate_plots, get_all_ids, reference_spectra_ids, get_spectrum_data

app = Flask(__name__)
db_file_path = 'app/database/microplastics_reference.db'  # Path to SQLite database

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            sample_peaks = process_and_plot_sample(file)
            results, best_match = calculate_similarity(sample_peaks)
            score = results[best_match]
            return render_template('index.html', results=results, best_match=best_match, score=score, sample_plot='sample_with_peaks.png')
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)