from dataset import SoccerNetClips, SoccerNetClipsTesting, SoccerNetClipsTrain

a = SoccerNetClipsTrain(store = True, chunk_size = 7, stride = 2)
print(a.__len__())
print(a.__getitem__(0))

