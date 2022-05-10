import SoccerNet

from SoccerNet.Downloader import SoccerNetDownloader, getListGames
import os
import numpy as np

path = "/data-net/datasets/SoccerNetv2/videos_lowres"
mySoccerNetDownloader = SoccerNetDownloader(
    LocalDirectory=path)

mySoccerNetDownloader.password = "s0cc3rn3t"

# download LQ (224p) Videos
splits = ['train', 'valid', 'test', 'challenge']
for split in splits:
    for game in getListGames(split):
        print(game)
        audio1 = np.load(os.path.join(path, game, '1_audio.npy'))
        print(audio1.shape)
        audio2 = np.load(os.path.join(path, game, '2_audio.npy'))
        print(audio2.shape)
        if (audio1.shape[0] == 0) | (audio2.shape[0] == 0):
            print('Downloading game: ' + game)
            mySoccerNetDownloader.downloadGame(game, files = ["1_224p.mkv", "2_224p.mkv"], spl = split)
#mySoccerNetDownloader.downloadGame(files=["1_224p.mkv", "2_224p.mkv"], split=["train","valid","test","challenge"])

