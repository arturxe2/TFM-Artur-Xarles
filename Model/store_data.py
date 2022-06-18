from dataset import SoccerNetClips, SoccerNetClipsTesting, SoccerNetClipsTrain

a = SoccerNetClipsTrain(store = True, chunk_size = 7, stride = 3, split = ['train'])
print(a.__len__())
print(a.__getitem__(0))

