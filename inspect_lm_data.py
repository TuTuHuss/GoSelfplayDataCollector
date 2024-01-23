import pickle


with open('dataset.pkl', 'rb') as f:
    data = pickle.load(f)


idx = 0
while True:
    print(data[idx]['sentence'])
    print(data[idx]['answer'])
    input()
    idx += 1
