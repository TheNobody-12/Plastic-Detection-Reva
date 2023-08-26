from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import sqlite3

app = Flask(__name__)

@app.route('/bubblemap')
def index():
    # get the data from the sqlite database 
    conn = sqlite3.connect("object_detection.db")
    cursor = conn.cursor()
    # Fetch unique filenames and their count with their unique lat and long from the database
    cursor.execute("SELECT filename, COUNT(filename), latitude, longitude FROM object_detection_data GROUP BY filename, latitude, longitude")
    rows = cursor.fetchall()

    # Create a dataframe from the rows
    df = pd.DataFrame(rows, columns=['filename', 'Plastic_count', 'latitude', 'longitude'])
    fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', size='Plastic_count',
                            color='Plastic_count', color_continuous_scale='plasma',
                            zoom=18, mapbox_style='open-street-map')
    # add filename to the hover data
    fig.update_traces(hovertemplate='<b>%{text}</b><br>' +
                                    'Plastic Count: %{marker.size:,}<br>' +
                                    'Latitude: %{lat}<br>' +
                                    'Longitude: %{lon}<br>',
                        text=df['filename'])
    
    fig.update_layout(title='Bubble Map of Plastic Count in the Ocean')

    plot_div = fig.to_html(full_html=False)

    # return plot_div

    return render_template('map.html', plot_div=plot_div)

if __name__ == '__main__':
    app.run(debug=True,port=5100)
