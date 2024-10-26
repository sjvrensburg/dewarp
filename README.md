# dewarp

A Python package for document scanning and dewarping that corrects perspective distortion in images. It uses the excellent [DocScanner](https://github.com/fh2019ustc/DocScanner?tab=readme-ov-file) with [weights](https://drive.google.com/drive/folders/1W1_DJU8dfEh6FqDYqFQ7ypR38Z8c5r4D?usp=sharing) provided by the original authors. Please cite their work if you find this useful:

```bib
@inproceedings{feng2021doctr,
  title={DocTr: Document Image Transformer for Geometric Unwarping and Illumination Correction},
  author={Feng, Hao and Wang, Yuechen and Zhou, Wengang and Deng, Jiajun and Li, Houqiang},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={273--281},
  year={2021}
}

@inproceedings{feng2022docgeonet,
  title={Geometric Representation Learning for Document Image Rectification},
  author={Feng, Hao and Zhou, Wengang and Deng, Jiajun and Wang, Yuechen and Li, Houqiang},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2022}
}

@article{feng2021docscanner,
  title={DocScanner: robust document image rectification with progressive learning},
  author={Feng, Hao and Zhou, Wengang and Deng, Jiajun and Tian, Qi and Li, Houqiang},
  journal={arXiv preprint arXiv:2110.14968},
  year={2021}
}
```

**NON-COMMERCIAL USAGE ONLY UNLESS YOU OBTAIN PRIOR PERMISSION FROM THE AUTHORS OF [DocScanner](https://github.com/fh2019ustc/DocScanner?tab=readme-ov-file) with [weights](https://drive.google.com/drive/folders/1W1_DJU8dfEh6FqDYqFQ7ypR38Z8c5r4D?usp=sharing)!**

## Installation

```bash
# Using pip
pip install git+https://github.com/sjvrensburg/dewarp.git

# To add the library to a Poetry project
poetry add git+https://github.com/sjvrensburg/dewarp.git
```

## Usage

### As a Command-Line Tool

```bash
# Process a directory of images
dewarp --input_dir ./distorted --output_dir ./rectified

# Use specific device
dewarp --input_dir ./distorted --output_dir ./rectified --device cuda
```

### As a Library

```python
from dewarp import DocumentScannerPipeline

# Initialize the pipeline
pipeline = DocumentScannerPipeline()

# Process a single image
pipeline.process_image(
    image_path="input.jpg",
    output_path="output.jpg"
)

# Process a directory of images
pipeline.process_directory(
    input_dir="./distorted",
    output_dir="./rectified"
)
```

## Licence

Unfortunately, the authors of [DocScanner](https://github.com/fh2019ustc/DocScanner?tab=readme-ov-file) and its [weights](https://drive.google.com/drive/folders/1W1_DJU8dfEh6FqDYqFQ7ypR38Z8c5r4D?usp=sharing) did not provide a licence. For commercial usage, you should contact Professor Wengang Zhou ([zhwg@ustc.edu.cn](zhwg@ustc.edu.cn)) and Hao Feng ([haof@mail.ustc.edu.cn](haof@mail.ustc.edu.cn)) 
