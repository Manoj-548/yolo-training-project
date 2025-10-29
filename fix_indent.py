with open('datasets/random_split_dataset/train_numpy.py', 'r') as f:
    content = f.read()
with open('datasets/random_split_dataset/train_numpy.py', 'w') as f:
    f.write(content.replace('\t', '    '))
