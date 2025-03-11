from datasets import load_dataset

dataset = load_dataset('laolao77/ViRFT_CLS_fgvc_aircraft_4_shot')
print(dataset)

# 展示数据集的信息
print(dataset.info)