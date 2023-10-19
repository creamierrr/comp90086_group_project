## COMP90086 Computer Vision Group Project

#### Similarity Detection Models Through Human Lens

> Computer Vision's progress has produced advanced Deep Architectures whose performance on rigid tasks like classification beats humans, but human vision systems and the way humans process visual signals are much richer than the traditional objectives Computer Vision models train to achieve. In this work, the way humans detect similarity is modelled using a Siamese Network with Triplet Learning, on the Totally-Looks-Alike data-set which is curated with image pairs of unrelated objects which absurdly look similar. This work proposes a model with 61.7% top2-accuracy over 20 candidate images, building towards computer vision machine learning with greater human qualities with a variety of applications in robotics and digital content monitoring.

#### Group Members

- **Name:** Lang (Ron) Chen **Student ID:** 1181506 **Email:** Lachen1@student.unimelb.edu.au
- **Name:** Anhui Situ **Student ID:** 1173962 **Email:** asitu@student.unimelb.edu.au


### Getting Started

> These instructions will help you get a copy of the project up and running on your local machine for development and testing purposes.

#### Prerequisites

> Before you begin, ensure you have met the following requirements:

* **Python 3.6** or higher installed on your system.

* **Jupyter Notebook** installed on your system.

* **PyTorch** installed. You can install it using pip:
```bash
pip install torch torchvision
```

* Additional Python packages listed in the `requirements.txt` file. You can install them using pip:
```bash
pip install -r requirements.txt
```
* Now that you've set up the environment and installed the necessary packages, you're ready to run the project.

#### Running the Code 

> If you want to train models or reproduce our results, follow these steps:

* Prepare your dataset by following the instructions in the dataset directory (`COMP90086_2023_TLLdataset 2/`). Make sure that the dataset is in the same directory as `main.ipynb`.

    * The directory structure might look something like this:

        ```css
        project_directory/
        │
        ├── main.ipynb
        │
        ├── requirements.txt
        │
        ├── COMP90086_2023_TLLdataset 2/
        │   │
        │   ├── test/
        │   │   │
        │   │   ├── ...
        │   │
        │   ├── train/
        │   │   │
        │   │   ├── ...
        │   │
        │   ├── train.csv
        │   │   
        │   │   
        │   └── ...
        ```

* Open `main.ipynb` and execute the cells sequentially to run the project. 

    * The first `Load the libraries and network architecture` section imports the necessary packages and defines some helper functions.
    * The second `Load the data and pre-process the data` section loads the images and pre-process the images.
    * The following sections define the models and train the models.
      * Our **final model** is `Experiment 2: Semi-hard negative mining` using `DenseNet201` as the backbone, which can be located under the `Siamese Constrative` section.
      * Other models denote different experiments we have done. 
      * You can train the model by executing the cells under the corresponding section.

* Upon successful training process, the trained models will be saved in the `state/` directory. Test predictions will be output to a corresponding csv file in the project directory. You can load the trained models and use them for inference.
```python
# load the trained model
cnn_siamise_triplet.load()
```

* The **final test prediction** is saved in `final_prediction.csv` in the project directory. We submitted it to Kaggle to get the final score.
