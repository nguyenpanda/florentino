# 🤺florentino🤺

---
[![License: Apache 2.0](https://img.shields.io/github/license/saltstack/salt)](https://opensource.org/license/apache-2-0)
![PyPI - Version](https://img.shields.io/pypi/v/florentino)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/florentino)

- Have you ever felt horny implementing AI models from scratch using only Numpy? No?

## How to Install

---

Installing the necessary libraries:

```bash
pip install -r requirements.txt
```

## Available models

--- 

<details>
  <summary>Click to expand!</summary>

1. **Machine Learning Model**
    - Linear Model `florentino.linear_model`
        - Linear regression `LinearRegression`

2. **Deep Learning Model**
    - Neural Network `florentino.nn`
        - Fully connected layer `Dense`
        - Softmax `Softmax`

</details>


<details>
   <summary>Run the following code to view all available models</summary>

```bash
python -c "
import florentino as flo

yellow, magenta, reset = '\033[1;93m', '\033[1;95m', '\033[0m'
header = ' All modules in florentino '
print(yellow + f'{header:=^50}' + reset)
for submodule in flo.__all__:
    exec(f'from florentino import {submodule} as submodule')
    print(magenta + submodule.__name__ + reset)
    for class_name in submodule.__all__:
        print(f'\\t - {class_name}')
    del submodule
print(yellow + 50 * '=' + reset)
"
```

</details>

## License

---

This project is licensed under the Apache Software License.

## Contact

---

- **nguyenpanda**: [hatuongnguyen0107@gmail.com](hatuongnguyen0107@gmail.com)
- **restingkiwi**: [khoana2003@gmail.com](khoana2003@gmail.com)
