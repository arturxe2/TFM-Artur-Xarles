path = '/data-net/datasets/SoccerNetv2/data_split/'
with open(path + 'train.txt') as f:
    lines = f.readlines()

for line in lines:
    print(line)
