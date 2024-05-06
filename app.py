from flask import Flask, render_template, request, send_from_directory
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sentence_transformers import SentenceTransformer
import json
import os
import matplotlib.pyplot as plt
from flask import send_file
import io
import plotly.graph_objs as go


app = Flask(__name__)

df = pd.read_excel('geeta_new.xlsx')

df['Shloka Number'] = df.groupby('Chapter').cumcount() + 1

ds = pd.read_excel('cosine_similarity_new.xlsx')
adidevananda = pd.read_excel('senti_adi.xlsx')
purohit = pd.read_excel('senti_purohit.xlsx')
sivananda = pd.read_excel('senti_siva.xlsx')
gandhi = pd.read_excel('senti_gandhi.xlsx')

with open('Adidevananda_vec.pkl', 'rb') as f:
    Adidevananda_vec = pickle.load(f)

model = SentenceTransformer('all-mpnet-base-v2')

def get_open_port():
    temp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    temp_socket.bind(('localhost', 0))  # Bind to port 0
    addr, port = temp_socket.getsockname()
    temp_socket.close()
    return port

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    user_input = request.form['user_input']
    translation_type = request.form['translation_type']
    top_count = int(request.form['top_count'])  
    
    user_vector = model.encode([user_input])[0]
    similarity_scores = cosine_similarity([user_vector], Adidevananda_vec)
    
    top_indices = similarity_scores.argsort()[0][-top_count:][::-1]

    similar_shlokas = []
    for index in top_indices:
        chapter = df.iloc[index]['Chapter']
        shloka_number = df.iloc[index]['Shloka Number'] 
        
        if translation_type == 'Original':
            verse = df.iloc[index]['Shloka']
        elif translation_type == 'Adidevananda':
            verse = df.iloc[index]['English Translation By Swami Adidevananda']
        elif translation_type == 'Sivananda':
            verse = df.iloc[index]['English Translation By Swami Sivananda']
        elif translation_type == 'Gandhi':
            verse = df.iloc[index]['English Translation By Mahatama Gandhi']
        elif translation_type == 'Purohit':
            verse = df.iloc[index]['English Translation By Shri Purohit Swami']
        else:
            verse = 'Invalid translation type'

        similarity_percentage = similarity_scores[0][index] * 100
        similar_shlokas.append((chapter, shloka_number, verse, round(similarity_percentage, 2)))

    return render_template('results.html', user_input=user_input, similar_shlokas=similar_shlokas, translation_type=translation_type)

@app.route('/analyze_semantic_similarity', methods=['POST'])
def analyze_semantic_similarity():
    chapter = request.form['chapter']
    author_pair_col_name = request.form['author_pair']
    
    if chapter == '19':
        top_similar_df = pd.read_excel('max_sum2.xlsx')
        max_value_row = top_similar_df.loc[top_similar_df[author_pair_col_name].idxmax()]
        most_similar_verse = max_value_row['Shloka']
        shloka_number = max_value_row['Shloka Number']
        ch=max_value_row['Chapter']
        mean_cosine_similarity_df = pd.read_excel('mean_cosine_similarity.xlsx')
        trace = go.Scatter(x=mean_cosine_similarity_df['Chapter'], y=mean_cosine_similarity_df[author_pair_col_name], mode='lines', name='Graph showing mean values of cosine scores across all 18 chapters')
        layout = go.Layout(title='Graph showing mean values of cosine scores across all 18 chapters', xaxis=dict(title='Chapters'), yaxis=dict(title='Similarity'))
        fig = go.Figure(data=[trace], layout=layout)
        
        graph_html = fig.to_html(full_html=False, default_height=500, default_width=700)
        
          
        mean_cosine_similarity_value = mean_cosine_similarity_df[author_pair_col_name].mean()
        
    else:
        
        filtered_df = ds[ds['Chapter'] == int(chapter)].copy()  
        filtered_df.reset_index(drop=True, inplace=True)
        
        top_similar_df = pd.read_excel('max_sum2.xlsx')  
        top_similar_chapter = top_similar_df[top_similar_df['Chapter'] == int(chapter)]
        most_similar_shloka_index = top_similar_chapter.index[0]
        most_similar_verse = top_similar_df.loc[most_similar_shloka_index, 'Shloka']
        shloka_number = top_similar_chapter.loc[most_similar_shloka_index, 'Shloka Number'] 
        ch=chapter
        trace = go.Scatter(x=filtered_df.index, y=filtered_df[author_pair_col_name], mode='lines', name=author_pair_col_name)
        layout = go.Layout(title=f'Chapter {chapter} - {author_pair_col_name}', xaxis=dict(title='Verse', tickvals=[], ticktext=[]), yaxis=dict(title='Similarity'))
        fig = go.Figure(data=[trace], layout=layout)
        
        graph_html = fig.to_html(full_html=False, default_height=500, default_width=700)
        
        mean_cosine_similarity_df = pd.read_excel('mean_cosine_similarity.xlsx')  
        mean_cosine_similarity_value = mean_cosine_similarity_df.loc[int(chapter)-1, author_pair_col_name]
    
    return render_template('semantic_similarity.html',ch=ch, chapter=chapter, shloka_number=shloka_number, author_pair_col_name=author_pair_col_name, graph_html=graph_html, most_similar_verse=most_similar_verse, mean_cosine_similarity=mean_cosine_similarity_value)

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    chapter = int(request.form['chapter'])

    translations = [adidevananda, purohit, sivananda, gandhi]
    translation_names = ['Adidevananda', 'Purohit', 'Sivananda', 'Gandhi']
    data = []

    if chapter==19:
        for translation, name in zip(translations, translation_names):
            counts = translation['label'].value_counts()
            trace = go.Bar(x=counts.index, y=counts.values, name=name)
            data.append(trace)
        layout = go.Layout(
            title='Overall Sentiment Analysis',
            xaxis=dict(title='Emotions'),
            yaxis=dict(title='Verse Count'),
            barmode='group'
        )
    else:
        for translation, name in zip(translations, translation_names):
            counts = translation[translation['Chapter'] == chapter]['label'].value_counts()
            trace = go.Bar(x=counts.index, y=counts.values, name=name)
            data.append(trace)
        layout = go.Layout(
            title=f'Sentiment Analysis for Chapter {chapter}',
            xaxis=dict(title='Emotions'),
            yaxis=dict(title='Verse Count'),
            barmode='group'
        )

    
    fig = go.Figure(data=data, layout=layout)
    graph_html = fig.to_html(full_html=False, default_height=500, default_width=700)

    return render_template('sentiment.html', chapter=chapter, graph_html=graph_html)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

