import SoccerNet

from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(
    LocalDirectory="/data-net/datasets/SoccerNetv2/videos_lowres")

mySoccerNetDownloader.password = 's0cc3rn3t'

# download LQ (224p) Videos
mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train","valid","test","challenge"])

