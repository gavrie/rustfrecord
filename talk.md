---
marp: true
---

# From TensorFlow to PyTorch
## With some help from Rust

<br>

Gavrie Philipson
Rusty Bits Software Ltd.
June 2025

---

# About Me

## Gavrie Philipson

- Rust, Python, Cloud, Backend, DevOps, and more.
- Bootstrapping software development teams: Training, mentoring, and hiring
- Consulting to startup companies on software development and architecture

<br>

Rusty Bits Software Ltd.
https://rustybits.io
gavrie@rustybits.io

---

# About You

---

# Using Rust to improve Python

<br>
<br>
<br>
<br>
<br>

![bg 40%](images/python-logo.png)
![bg 60%](images/ferris.png)
![bg 50%](images/iron-oxide.jpg)

<br>
<br>

[Astral](https://astral.sh)
[PyO<sub>3</sub>](https://github.com/PyO3)

---

# The Mission

- Port ML model from TensorFlow to PyTorch
- Lots of training data in `TFRecord` format

---

# The `TFRecord` format

- A sequence of `HashMap<String, Vec<T>>`
- `where T: u8 | f32 | i64`
- Serialized with `protobuf`

---

# `TFRecord` Example

```python
[
  {
    "label": "cat",
    "image/shape": [320, 200, 3],
    "image/encoded": [0x12, 0x34, 0x56, ...],
  },
  {
    "label": "dog",
    "image/shape": [320, 200, 3],
    "image/encoded": [0x78, 0x9a, 0xbc, ...],
  },
]
```

---

# The Constraints

- Dependencies (look at venv size)
- Performance: Keep GPUs busy
- Ease of use for Python devs

---

# Challenge: Getting Test Data

- No access to the original data
- Vibe code [some Python](tf_example/main.py) to generate test data!

---

# Playing on Rust's strengths

- Designing with types
- Dive into the [Rust implementation](tfrecord_reader/src/lib.rs) and [`tests.rs`](tfrecord_reader/src/tests.rs)

---

# The End Result

- `pip install rustfrecord` 
- [`test_rustfrecord.py`](test_rustfrecord.py)
- [`src/lib.rs`](src/lib.rs)

---

# Getting the Code

https://pypi.org/project/rustfrecord/
https://github.com/gavrie/rustfrecord