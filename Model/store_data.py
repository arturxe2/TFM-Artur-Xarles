from dataset import SoccerNetClips, SoccerNetClipsTesting, SoccerNetClipsTrain

a = SoccerNetClipsTrain(store = True, chunk_size = 4, stride = 2, split = ['train', 'valid', 'test'])
print(a.__len__())
print(a.__getitem__(0))

