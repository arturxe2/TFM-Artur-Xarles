import SoccerNet

from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(
    LocalDirectory="/data-net/datasets/SoccerNetv2/videos_lowres")

mySoccerNetDownloader.password = input("Password for videos?:\n")

# download LQ (224p) Videos
mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train","valid","test","challenge"])

