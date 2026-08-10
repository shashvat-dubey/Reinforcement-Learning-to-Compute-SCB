import pickle

with open("labelled_dataset.pkl", "rb") as f:
    data = pickle.load(f)


print(data[1])