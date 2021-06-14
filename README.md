# ort_inference

ONNX Runtime Inferencing.


## Installation

```python
git clone --recurse-submodules https://github.com/ethen8181/ort_inference.git

python3 setup.py install
```

## Examples

```python
# load the onnx model for sequence classification task
onnx_model_path = "text_classification.onnx"
intra_op_num_threads = 4
ort_inference = SequenceClassificationOrtInference(
    onnx_model_path,
    intra_op_num_threads
)

# we can directly perform prediction on a dynamic sequence length input_ids
input_ids = [
    [101, 3183, 2079, 2017, 2293, 1996, 2087, 1998, 2339, 1029,
     102, 3183, 2079, 2017, 2293, 2087, 1998, 2339, 1029, 102]
]

ort_inference.batch_predict(input_ids)
# output: [[-1.2110593,  1.7904084]]
```

Please refer to `examples` folder for more elaborated examples.
