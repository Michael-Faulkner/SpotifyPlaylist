import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util
import pandas as pd
import copy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import random

def credentialsetup(client_id, secret, username, redirect_url):
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=secret) 
    spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    scope = 'user-library-read playlist-read-private user-top-read playlist-modify-public'
    token = util.prompt_for_user_token(username, scope,client_id, secret, redirect_url)
    if token:
        spotify = spotipy.Spotify(auth=token)
    else:
        print("Can't get token for", username)
        
    return spotify
    
    
    
def get_playlist_ids(spotify, good_playlist = 'Python Test 1', bad_playlist = 'Python Test 0'):
    y = spotify.current_user_playlists()['items']
    playlists = pd.DataFrame(y)
    playlists = playlists[['name','id']]

    print("Grabbing songs from liked playlist")
    good_playlist_id = playlists[playlists['name'] == good_playlist]
    good_playlist_id = good_playlist_id.iloc[0,1]

    print("Grabbing songs from dislikde playlist")
    bad_playlist_id = playlists[playlists['name'] == bad_playlist]
    bad_playlist_id = bad_playlist_id.iloc[0,1]
    
    return good_playlist_id, bad_playlist_id


def get_song_attributes(spotify, good_playlist_id, bad_playlist_id):
    print("Grabbing song attributes from liked playlist")

    good_uncleaned = []
    for i in range(10):
        try:
            good_uncleaned.append(pd.DataFrame(spotify.playlist_tracks(good_playlist_id, offset = 100 * i)['items'])['track'])
        except:
            break
    good_uncleaned_tracks = pd.concat(good_uncleaned)
    good_uncleaned_tracks = good_uncleaned_tracks.reset_index(drop = True)
    good_track_ids = []

    for i in range(len(good_uncleaned_tracks)):
        song_id = good_uncleaned_tracks[i]['id']
        good_track_ids.append(song_id)
    good_feats = []

    for i in range(10):

        try:
            good_feats.append(pd.DataFrame(spotify.audio_features(good_track_ids[i*100:(i+1)*100])))
        except:
            print(i)
            good_feats.append(pd.DataFrame(spotify.audio_features(good_track_ids[i*100:])))
            break

    good_audio_feats = pd.concat(good_feats)
    good_audio_feats = good_audio_feats.drop(0, axis = 1)
    good_audio_feats = good_audio_feats.reset_index(drop = True)
    good_audio_feats = good_audio_feats.dropna()
    good_audio_feats['target'] = 1


    print("Grabbing song attributes from disliked playlist")
    bad_uncleaned  = []

    for i in range(10):
        try:
            bad_uncleaned.append(pd.DataFrame(spotify.playlist_tracks(bad_playlist_id, offset = 100*i)['items'])['track'])
        except:
            break

    bad_uncleaned_tracks = pd.concat(bad_uncleaned)

    bad_uncleaned_tracks = bad_uncleaned_tracks.reset_index(drop = True)
    bad_track_ids = []

    for i in range(len(bad_uncleaned_tracks)):
        song_id = bad_uncleaned_tracks[i]['id']
        bad_track_ids.append(song_id)
    bad_feats = []
    for i in range(10):

        try:
            bad_feats.append(pd.DataFrame(spotify.audio_features(bad_track_ids[i*100:(i+1)*100])))
        except:
            print(i)
            bad_feats.append(pd.DataFrame(spotify.audio_features(bad_track_ids[i*100:])))
            break

    bad_audio_feats = pd.concat(bad_feats)
    bad_audio_feats = bad_audio_feats.drop(0, axis = 1)
    bad_audio_feats = bad_audio_feats.reset_index(drop = True)
    bad_audio_feats = bad_audio_feats.dropna()
    bad_audio_feats['target'] = 0
    
    return good_audio_feats, bad_audio_feats, good_track_ids

def model_building(good_audio_feats, bad_audio_feats):
    print("Combining liked and disliked attributes and creating test and training data sets")
    all_audio_feats = pd.concat([good_audio_feats, bad_audio_feats])
    all_audio_feats = all_audio_feats.reset_index(drop = True)

    train, test = train_test_split(all_audio_feats, test_size = 0.2)
    features = ["danceability", "loudness", "valence", "energy", "instrumentalness", "acousticness", "key", "speechiness", 'liveness',"tempo"]

    x_train = train[features]
    y_train = train["target"]
    x_test = test[features]
    y_test = test["target"]

    ss = StandardScaler()
    x_train_scale = ss.fit_transform(x_train)
    x_test_scale = ss.transform(x_test)

    print("Training KNN Model")
    knn = KNeighborsClassifier(23, weights = 'uniform', metric = 'manhattan')
    knn.fit(x_train_scale, y_train)
    knn_pred = knn.predict(x_test_scale)
    score = accuracy_score(y_test, knn_pred) * 100
    print("Accuracy using Knn: ", round(score, 1), "%")

    print("Training Random Forest Model")
    clf = RandomForestClassifier(n_estimators = 427, min_samples_leaf=2, min_samples_split=2, max_depth = 10, max_features='log2', bootstrap=False)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    score = accuracy_score(y_test, y_pred) * 100
    print("Accuracy using Random Forest: ", round(score, 1), "%")
    
    return knn, clf, features
def potential_songs(spotify, good_track_ids, features):
    print("Creating list of potential songs based on liked songs")
    
    test_ids = []
    for i in range(len(good_track_ids)):
        x = spotify.recommendations(seed_artists=None, seed_genres=None, seed_tracks=[good_track_ids[i]], limit=3)
        for j in range(len(x['tracks'])):
            s_id = x['tracks'][j]['id']
            test_ids.append(pd.DataFrame(spotify.audio_features(s_id)))

    y = pd.concat(test_ids)
    y = y.reset_index(drop = True)
    z2 = y.drop_duplicates()
    song_test_features = z2[features]
    
    return song_test_features, z2

def model_prediction(song_test_features,z2, knn, clf, features):
    print("Predicting song outcome based on KNN and Random Forest models")
    test_song_list = []
    z2['result_rf'] = list(clf.predict(song_test_features))
    z2['result_knn'] = list(knn.predict(song_test_features))
    result = z2[features + ['result_rf','result_knn','id']]
    good_songs = result[(result['result_rf'] == 1) & result['result_knn'] == 1]
    test_song_list.append(good_songs)
    print(len(good_songs))
    return test_song_list, good_songs

def playlist_generation(spotify, good_songs, test_song_list, username, songs_in_playlist = 50):

    print("Creating New Playlist")
    name = 'ML'
    spotify.user_playlist_create(username, name, public=True, collaborative=False, description='')
    for item in spotify.current_user_playlists()['items']:
        if item['name'] == name:
            new_playlist_id = item['id']
    good_songs = good_songs.sample(songs_in_playlist)
    print("Adding Songs")        
    songs_by_100 = int(np.ceil(len(good_songs)/100))
    for i in range(songs_by_100):
        if i < (songs_by_100 - 1):
            songs_to_add = list(good_songs['id'][i*100:100*(i+1)])
            spotify.playlist_add_items(new_playlist_id, songs_to_add)
        else:
            songs_to_add = list(good_songs['id'][i*100:])
            spotify.playlist_add_items(new_playlist_id, songs_to_add)
    print("Finished")
    
    
    
def main():
    client_id = open('client.txt').readlines()[0]
    secret = open('secret.txt').readlines()[0]
    username = open('username.txt').readlines()[0]
    redirect_url = open('redirect.txt').readlines()[0]
    
    bad_playlist_name = input('Type the name of the playlist with songs you do NOT like, Default is Python Test 0: ')
    good_playlist_name = input("Type the name of the playlist with songs you DO like, Default is Python Test 1: ")
 
    
    spotify = credentialsetup(client_id = client_id, secret = secret, username = username, redirect_url = redirect_url)
    if bad_playlist_name:
        if good_playlist_name:
            good_playlist_id, bad_playlist_id = get_playlist_ids(spotify, good_playlist = good_playlist_name, bad_playlist = bad_playlist_name)
        else:
             good_playlist_id, bad_playlist_id = get_playlist_ids(spotify, bad_playlist = bad_playlist_name)
    elif good_playlist_name:
        good_playlist_id, bad_playlist_id = get_playlist_ids(spotify, good_playlist = good_playlist_name)
    else:
        good_playlist_id, bad_playlist_id = get_playlist_ids(spotify)
    good_audio_feats, bad_audio_feats, good_track_ids = get_song_attributes(spotify, good_playlist_id, bad_playlist_id)
    knn, clf,features = model_building(good_audio_feats, bad_audio_feats)
    song_test_features, z2 = potential_songs(spotify, good_track_ids, features)
    test_song_list, good_songs = model_prediction(song_test_features,z2, knn, clf, features)
    
    songs_in_playlist = input("How long do you want the output playlist to be? Default is 50: ")
    if songs_in_playlist:
        playlist_generation(spotify, good_songs, test_song_list, username, int(songs_in_playlist))
    else:
        playlist_generation(spotify, good_songs, test_song_list, username)
    
main()