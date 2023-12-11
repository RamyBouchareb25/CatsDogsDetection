# CatDog Emotion Detection with Computer Vision

This repository contains the code and resources for training a Computer Vision model using TensorFlow to detect emotions from sentences associated with photos of cats and dogs. The model is trained on a dataset consisting of labeled images of cats and dogs, allowing it to recognize and analyze emotional cues in pet images.

## Requirements

- Python 3.6 or higher
- TensorFlow 2.x

## Dataset

the Data set for this model are provided with tenserflow directly if you want to add you can put the images in a folder /data
run the command: 
```
pythin pre_process.py
```

## Training

To train the Computer Vision model, run the following command:

```bash
python train_model.py
```

This script will preprocess the images, train the model using a deep learning architecture, and save the trained model in the `models/` directory.

## Evaluation

Evaluate the model's performance using the test dataset:

```bash
python evaluate_model.py
```

This will generate metrics such as accuracy, precision, recall, and F1 score.

## Contributing

If you find any issues or have improvements to suggest, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

[//]: # (Replace [source-link] with the actual link to the dataset.)
