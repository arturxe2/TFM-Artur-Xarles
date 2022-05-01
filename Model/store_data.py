from dataset import SoccerNetClips, SoccerNetClipsTesting, SoccerNetClipsTrain

a = SoccerNetClipsTrain(store = True)
print(a.__len__())
#print(a.__getitem__([0, 1, 2, 50000]))

