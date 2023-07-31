import datasets

dataset = datasets.load_dataset(path="NeelNanda/pile-10k", split="train")
dataset.save_to_disk('.')

