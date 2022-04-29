from dataset import SoccerNetClips, SoccerNetClipsTesting, SoccerNetClipsTrain

a = SoccerNetClipsTrain(store = False)
print(a.__len__())
print(a.__getitem__([0, 2, 4, 6]))

