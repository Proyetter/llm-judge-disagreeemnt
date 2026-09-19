from datasets import load_dataset

ds = load_dataset("ScalerLab/JudgeBench")
print(ds)                          # shows splits/configs and row counts
print(ds[list(ds.keys())[0]][0])   # prints one full example so we see the fields
