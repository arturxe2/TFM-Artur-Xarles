'''
Code for TFM: Transformer-based Action Spotting for soccer videos

Code in this file downloads 
'''
import SoccerNet

from SoccerNet.Downloader import SoccerNetDownloader, getListGames
import os
import numpy as np
from moviepy.editor import VideoFileClip

path = "/data-local/data1-hdd/axesparraguera/vggish"
mySoccerNetDownloader = SoccerNetDownloader(
    LocalDirectory=path)

mySoccerNetDownloader.password = "s0cc3rn3t"

# download LQ (224p) Videos
splits = ['train', 'valid', 'test', 'challenge']
i = 0
for split in splits:
    print('Working on split: ' + split)
    for game in getListGames(split):
        i += 1
        print('Game ' + str(i) + ' of 500')
        print(game)
        
        exists = os.path.exists(os.path.join(path, game, '1_audio.npy'))
        
        if exists:
            print('Audio.npy file exists for this match with following shapes:')
            audio1 = np.load(os.path.join(path, game, '1_audio.npy'))
            print(audio1.shape)
            audio2 = np.load(os.path.join(path, game, '2_audio.npy'))
            print(audio2.shape)
            if (audio1.shape[0] == 0) | (audio2.shape[0] == 0):
                try:
                    print('Incorrect .npy shape')
                    print('Downloading game: ' + game)
                    mySoccerNetDownloader.downloadGame(game, files = ["1_224p.mkv", "2_224p.mkv"], spl = split)
                    
                    print('Extracting 1st half audio...')
                    clip = VideoFileClip(os.path.join(path, game, '1_224p.mkv'))
                    clip.audio.write_audiofile(os.path.join(path, game, '1_224.wav'))
                    os.remove(os.path.join(path, game, '1_224.mkv'))
                    
                    
                    print('Extracting 2nd half audio...')
                    clip = VideoFileClip(os.path.join(path, game, '2_224p.mkv'))
                    clip.audio.write_audiofile(os.path.join(path, game, '2_224.wav'))
                    os.remove(os.path.join(path, game, '2_224.mkv'))
                except:
                    print('Error extracting audio from mkv file\n\n añlkjsdfajsdflkasdfj')
                
            else:
                if os.path.exists(os.path.join(path, game, '1_224p.mp3')):
                    print('Removing mp3 audio 1st half')
                    os.remove(os.path.join(path, game, '1_224p.mp3'))
                if os.path.exists(os.path.join(path, game, '2_224p.mp3')):
                    print('Removing mp3 audio 2nd half')
                    os.remove(os.path.join(path, game, '2_224p.mp3'))
            
        else:
            try:
                print('There is not audio.npy file for this match')
                print('Downloading game: ' + game)
                mySoccerNetDownloader.downloadGame(game, files = ["1_224p.mkv", "2_224p.mkv"], spl = split)
                
                print('Extracting 1st half audio...')
                clip = VideoFileClip(os.path.join(path, game, '1_224p.mkv'))
                clip.audio.write_audiofile(os.path.join(path, game, '1_224.wav'))
                os.remove(os.path.join(path, game, '1_224.mkv'))
                
                
                print('Extracting 2nd half audio...')
                clip = VideoFileClip(os.path.join(path, game, '2_224p.mkv'))
                clip.audio.write_audiofile(os.path.join(path, game, '2_224.wav'))
                os.remove(os.path.join(path, game, '2_224.mkv'))
                
            except:
                print('Error extracting audio from mkv file\n\n añlkjsdfajsdflkasdfj')
#mySoccerNetDownloader.downloadGame(files=["1_224p.mkv", "2_224p.mkv"], split=["train","valid","test","challenge"])

