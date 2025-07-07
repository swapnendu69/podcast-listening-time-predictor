from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)
# Load the trained model
model = pickle.load(open('podcast.sav', 'rb'))

@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    Podcast_Name = str(request.form['Podcast_Name'])
    Episode_Length_minutes = float(request.form['Episode_Length_minutes'])
    Genre = str(request.form['Genre'])
    Host_Popularity_percentage = float(request.form['Host_Popularity_percentage'])
    Publication_Day = str(request.form['Publication_Day'])
    Publication_Time = str(request.form['Publication_Time'])
    Guest_Popularity_percentage = float(request.form['Guest_Popularity_percentage'])
    Number_of_Ads = int(request.form['Number_of_Ads'])
    Episode_Sentiment = str(request.form['Episode_Sentiment'])

    # Create a DataFrame
    input_data = pd.DataFrame([[Podcast_Name, Episode_Length_minutes, Genre, Host_Popularity_percentage, 
                                Publication_Day, Publication_Time, Guest_Popularity_percentage, 
                                Number_of_Ads, Episode_Sentiment]],
                              columns=['Podcast_Name', 'Episode_Length_minutes', 'Genre', 
                                       'Host_Popularity_percentage', 'Publication_Day', 
                                       'Publication_Time', 'Guest_Popularity_percentage', 
                                       'Number_of_Ads', 'Episode_Sentiment'])

    # Ensure categorical variables are properly encoded (if necessary)
    # Example: If you used label encoding, apply the same encoding here

    # Make prediction
    result = model.predict(input_data)[0]

    return render_template('index.html', **locals())

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=8080)
