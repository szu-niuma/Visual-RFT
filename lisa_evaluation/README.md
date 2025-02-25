## ViRFT for reasoning grounding

Download `lisa_test.json` and `lisa_val.json` at [here](https://huggingface.co/datasets/Zery/BS-Objaverse).

After training model, replace model path in `Qwen2_VL_lisa_infere.py` with your own ckpt.

```python
# Load Qwen2-VL-2B model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/path/to/your/checkpoint-498", torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"
).eval()

processor = AutoProcessor.from_pretrained("/path/to/your/checkpoint-498")
```

```shell
cd lisa_evaluation
bash Qwen2_VL_lisa_infere.sh
```
