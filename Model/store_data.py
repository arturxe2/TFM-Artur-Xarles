from dataset import SoccerNetClips, SoccerNetClipsTesting, SoccerNetClipsTrain

a = SoccerNetClipsTrain(store = True, chunk_size = 3, stride = 1, split = ['train'])
print(a.__len__())
print(a.__getitem__(0))

