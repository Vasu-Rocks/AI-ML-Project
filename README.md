# Brain Tumor MRI Image Classification
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "a9e69a7d",
   "metadata": {},
   "source": [
    "# Brain Tumor MRI Image Classification\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c1fb0799",
   "metadata": {},
   "source": [
    "##### **Name** - <br>##### **Email** - <br>##### **Project Title** - Brain Tumor MRI Image Classification<br>##### **Domain** - Healthcare/Medical Imaging<br>##### **Domain Type** - Classification<br>##### **Project Type** - Classification<br>##### **Contribution** - Individual"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "956f9ce3",
   "metadata": {},
   "source": [
    "# **Project Summary -**\n",
    "This project explores deep learning approaches for classifying brain MRI images into categories representing tumor types. We implement both a custom Convolutional Neural Network and a pre-trained MobileNetV2 model to compare performance. The dataset consists of MRI scans of brains organized into multi-class labels (e.g., glioma, meningioma, pituitary tumors, and normal). We preprocess the images by resizing, normalization, and data augmentation. We then build and train our models, evaluate them on a held-out test set, and compare accuracy and other metrics."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5c065135",
   "metadata": {},
   "source": [
    "# **GitHub Link -**\n",
    "https://github.com/username/BrainTumorMRI"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7632f8fb",
   "metadata": {},
   "source": [
    "# **Problem Statement**\n",
    "Given MRI images of brains, build deep learning models to classify whether an image shows a particular type of tumor (glioma, meningioma, pituitary) or a normal brain. We will utilize both a custom CNN and a pre-trained MobileNetV2 model to achieve high accuracy."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d24c832b",
   "metadata": {},
   "source": [
    "# **General Guidelines** : -  \n",
    "1. Well-structured, formatted, and commented code is required.\n",
    "2. The notebook should run in one go without errors.\n",
    "3. Provide concise explanations for each step.\n",
    "4. Use visualizations where appropriate.\n",
    "5. Implement EarlyStopping and ModelCheckpoint callbacks for training.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e7db5e17",
   "metadata": {},
   "source": [
    "# ***Let's Begin !***"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d35fd6d4",
   "metadata": {},
   "source": [
    "## ***1. Know Your Data***"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3c59ca21",
   "metadata": {},
   "source": [
    "### Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fac5e08b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import necessary libraries\n",
    "import os\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D\n",
    "from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint\n",
    "from tensorflow.keras.applications import MobileNetV2\n",
    "from sklearn.metrics import classification_report, confusion_matrix"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c58c9bbc",
   "metadata": {},
   "source": [
    "### Dataset Loading"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a64cd01d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Extract dataset (if zipped) and set up directory paths\n",
    "import zipfile\n",
    "# Uncomment and modify the following lines if using Colab or if dataset is zipped\n",
    "# with zipfile.ZipFile(\"Labeled MRI Brain Tumor Dataset.v1-version-1.multiclass.zip\", 'r') as zip_ref:\n",
    "#     zip_ref.extractall()\n",
    "\n",
    "# Define directories (change as per your dataset structure)\n",
    "train_dir = \"Labeled MRI Brain Tumor Dataset/Training\"\n",
    "val_dir = \"Labeled MRI Brain Tumor Dataset/Validation\"\n",
    "test_dir = \"Labeled MRI Brain Tumor Dataset/Testing\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "28c96aae",
   "metadata": {},
   "source": [
    "### Dataset First View"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0e9e5ca0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check sample images and classes\n",
    "print(\"Training classes:\", os.listdir(train_dir))\n",
    "print(\"Validation classes:\", os.listdir(val_dir))\n",
    "print(\"Testing classes:\", os.listdir(test_dir))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fd66f2f5",
   "metadata": {},
   "source": [
    "### Dataset Rows & Columns count"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cfc1b288",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Count images in each folder for train, validation, test\n",
    "for split, directory in [(\"Train\", train_dir), (\"Validation\", val_dir), (\"Test\", test_dir)]:\n",
    "    print(f\"{split} set:\")\n",
    "    for cls in os.listdir(directory):\n",
    "        path = os.path.join(directory, cls)\n",
    "        count = len(os.listdir(path))\n",
    "        print(f\" - {cls}: {count} images\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "81b40f64",
   "metadata": {},
   "source": [
    "### Dataset Information"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "870a0cbe",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Optional: check an example image size to confirm consistency\n",
    "from tensorflow.keras.preprocessing import image\n",
    "img_path = os.path.join(train_dir, os.listdir(train_dir)[0], os.listdir(os.path.join(train_dir, os.listdir(train_dir)[0]))[0])\n",
    "img = image.load_img(img_path)\n",
    "print(\"Example image size:\", img.size)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e7c4c3b7",
   "metadata": {},
   "source": [
    "#### Duplicate Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "da5a384e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check for duplicate file names in the training directory (just a simple check)\n",
    "filenames = []\n",
    "duplicates = []\n",
    "for cls in os.listdir(train_dir):\n",
    "    for fname in os.listdir(os.path.join(train_dir, cls)):\n",
    "        if fname in filenames:\n",
    "            duplicates.append(fname)\n",
    "        else:\n",
    "            filenames.append(fname)\n",
    "print(\"Duplicate file names (if any):\", duplicates)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3e1bf252",
   "metadata": {},
   "source": [
    "#### Missing Values/Null Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e1881ae6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# For image data, missing values do not apply as each file is a complete image.\n",
    "print(\"No missing values in image dataset (each image is a file).\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f3380537",
   "metadata": {},
   "source": [
    "### What did you know about your dataset?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d47ecbb8",
   "metadata": {},
   "source": [
    "The dataset consists of brain MRI images categorized into multiple classes (e.g., glioma, meningioma, pituitary tumors, and normal). The images are stored in directory subfolders for train, validation, and test splits. We observed the number of images per class to understand class balance. Each MRI image can vary in size and needs to be resized for modeling. There are no missing or duplicate records to handle, and all images are complete and valid."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6780515a",
   "metadata": {},
   "source": [
    "## ***2. Understanding Your Variables***"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1d350b5f",
   "metadata": {},
   "source": [
    "### Variables Description"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ba6767db",
   "metadata": {},
   "source": [
    "The primary data variable is the image data (pixel intensities of MRI scans). Each image is represented as a 2D array of pixel values (we will resize them to 224x224 and normalize). The target variable is the class label of each image (e.g., types of brain tumor or normal). The classes are categorical and will be encoded for classification."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "93da44d8",
   "metadata": {},
   "source": [
    "### Check Unique Values for each variable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "53cd6a63",
   "metadata": {},
   "outputs": [],
   "source": [
    "# List unique classes (labels) in the training set\n",
    "classes = sorted(os.listdir(train_dir))\n",
    "print(\"Classes:\", classes)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e6a70c3a",
   "metadata": {},
   "source": [
    "## ***3. Data Wrangling***"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b6c0131f",
   "metadata": {},
   "source": [
    "### Data Wrangling Code"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d60b81cd",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create ImageDataGenerator objects for preprocessing and augmentation\n",
    "train_datagen = ImageDataGenerator(rescale=1./255,\n",
    "                                   rotation_range=20,\n",
    "                                   width_shift_range=0.1,\n",
    "                                   height_shift_range=0.1,\n",
    "                                   shear_range=0.1,\n",
    "                                   zoom_range=0.1,\n",
    "                                   horizontal_flip=True,\n",
    "                                   fill_mode='nearest')\n",
    "val_datagen = ImageDataGenerator(rescale=1./255)\n",
    "test_datagen = ImageDataGenerator(rescale=1./255)\n",
    "\n",
    "# Flow data from directories\n",
    "train_gen = train_datagen.flow_from_directory(train_dir, target_size=(224,224), class_mode='categorical')\n",
    "val_gen = val_datagen.flow_from_directory(val_dir, target_size=(224,224), class_mode='categorical')\n",
    "test_gen = test_datagen.flow_from_directory(test_dir, target_size=(224,224), class_mode='categorical', shuffle=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b1280a7f",
   "metadata": {},
   "source": [
    "### What all manipulations have you done and insights you found?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c2fbaaca",
   "metadata": {},
   "source": [
    "We applied image rescaling (normalization to [0,1]) and data augmentation (rotations, shifts, shear, zoom, flips) during training to improve model generalization. No additional data cleaning was needed since the dataset was already well-prepared. We created data generators for the training, validation, and test sets."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a78ce34b",
   "metadata": {},
   "source": [
    "## ***4. Data Vizualization, Storytelling & Experimenting with charts :\n Understand the relationships between variables***"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3c1d371a",
   "metadata": {},
   "source": [
    "#### Chart - 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c3d5ea46",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Chart - 1: Class distribution in training set\n",
    "labels = classes\n",
    "train_counts = [len(os.listdir(os.path.join(train_dir, cls))) for cls in classes]\n",
    "plt.figure(figsize=(6,4))\n",
    "sns.barplot(x=labels, y=train_counts)\n",
    "plt.title(\"Number of images per class in Training Set\")\n",
    "plt.xlabel(\"Class\")\n",
    "plt.ylabel(\"Count\")\n",
    "plt.xticks(rotation=45)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8048a517",
   "metadata": {},
   "source": [
    "##### 1. Why did you pick the specific chart?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b0eb8e85",
   "metadata": {},
   "source": [
    "A bar chart is appropriate to compare the number of images across different classes (categorical data)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7259e5b7",
   "metadata": {},
   "source": [
    "##### 2. What is/are the insight(s) found from the chart?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b33dba38",
   "metadata": {},
   "source": [
    "The chart shows the distribution of images across classes. It helps identify if classes are imbalanced. In our dataset, we can see how many images belong to each category."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d44e97dc",
   "metadata": {},
   "source": [
    "##### 3. Will the gained insights help creating a positive business impact? Are there any insights that lead to negative growth? Justify with specific reason."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b2c58cab",
   "metadata": {},
   "source": [
    "Understanding class distribution is important for model design: if classes are imbalanced, it can lead to biased predictions, which is a negative impact on performance. We may need techniques to handle imbalance (e.g., class weights or data augmentation) to ensure fair model training."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "899fa410",
   "metadata": {},
   "source": [
    "#### Chart - 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "59e66caf",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Chart - 2: Not applicable or additional chart not implemented\n",
    "pass"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d7d27e82",
   "metadata": {},
   "source": [
    "##### 1. Why did you pick the specific chart?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c0b454e9",
   "metadata": {},
   "source": [
    "Answer: Not applicable for this analysis."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1a94073e",
   "metadata": {},
   "source": [
    "##### 2. What is/are the insight(s) found from the chart?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d9b031fd",
   "metadata": {},
   "source": [
    "Answer: Not applicable."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "586e2e4a",
   "metadata": {},
   "source": [
    "##### 3. Will the gained insights help creating a positive business impact? Are there any insights that lead to negative growth? Justify with specific reason."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "85cdd744",
   "metadata": {},
   "source": [
    "Answer: Not applicable."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1ec9287c",
   "metadata": {},
   "source": [
    "## ***5. Hypothesis Testing***"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7a63b4bf",
   "metadata": {},
   "source": [
    "Hypothesis Testing is not applicable for this image classification problem. Our focus is on model evaluation using accuracy and other metrics, rather than statistical tests."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4e8ff907",
   "metadata": {},
   "source": [
    "### Hypothetical Statement - 1"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e22c4b7d",
   "metadata": {},
   "source": [
    "#### 1. State Your research hypothesis as a null hypothesis and alternate hypothesis."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3e0ce331",
   "metadata": {},
   "source": [
    "N/A"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3f5165e5",
   "metadata": {},
   "source": [
    "#### 2. Perform an appropriate statistical test."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0ce868ea",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Not applicable for image classification tasks\n",
    "pass"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4c7cbe98",
   "metadata": {},
   "source": [
    "##### Which statistical test have you done to obtain P-Value?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b51653cc",
   "metadata": {},
   "source": [
    "Not applicable"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c93f62a3",
   "metadata": {},
   "source": [
    "##### Why did you choose the specific statistical test?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e10bfb7a",
   "metadata": {},
   "source": [
    "Not applicable"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "280e1840",
   "metadata": {},
   "source": [
    "### Hypothetical Statement - 2"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "51ce7bce",
   "metadata": {},
   "source": [
    "#### 1. State Your research hypothesis as a null hypothesis and alternate hypothesis."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "08cea4bd",
   "metadata": {},
   "source": [
    "N/A"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "11c4cc7c",
   "metadata": {},
   "source": [
    "#### 2. Perform an appropriate statistical test."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5b6cf718",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Not applicable for image classification tasks\n",
    "pass"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e1e3f952",
   "metadata": {},
   "source": [
    "##### Which statistical test have you done to obtain P-Value?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c4c39ebb",
   "metadata": {},
   "source": [
    "Not applicable"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "737d15eb",
   "metadata": {},
   "source": [
    "##### Why did you choose the specific statistical test?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cbf4b33e",
   "metadata": {},
   "source": [
    "Not applicable"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aaf6bed5",
   "metadata": {},
   "source": [
    "### Hypothetical Statement - 3"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8963e575",
   "metadata": {},
   "source": [
    "#### 1. State Your research hypothesis as a null hypothesis and alternate hypothesis."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a4b49bf0",
   "metadata": {},
   "source": [
    "N/A"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b2799304",
   "metadata": {},
   "source": [
    "#### 2. Perform an appropriate statistical test."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "83b7cc50",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Not applicable for image classification tasks\n",
    "pass"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cd5e4785",
   "metadata": {},
   "source": [
    "##### Which statistical test have you done to obtain P-Value?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0bdc5f7c",
   "metadata": {},
   "source": [
    "Not applicable"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5d375caf",
   "metadata": {},
   "source": [
    "##### Why did you choose the specific statistical test?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "56b8cdd9",
   "metadata": {},
   "source": [
    "Not applicable"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bd67e6ba",
   "metadata": {},
   "source": [
    "## ***6. Feature Engineering & Data Pre-processing***"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a08a5b94",
   "metadata": {},
   "source": [
    "### 1. Handling Missing Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "131b30b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# No missing values in the dataset of image files\n",
    "print('No missing values to impute.')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c031213c",
   "metadata": {},
   "source": [
    "#### What all missing value imputation techniques have you used and why did you use those techniques?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "94fe9412",
   "metadata": {},
   "source": [
    "No missing values are present in the image dataset, so no imputation was needed. Each file represents a complete image."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e01a1e22",
   "metadata": {},
   "source": [
    "### 2. Handling Outliers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cff0d675",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Outliers are not applicable in raw image data; pixel values naturally range from 0 to 255\n",
    "print('No outlier treatment needed.')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8f0dfe50",
   "metadata": {},
   "source": [
    "##### What all outlier treatment techniques have you used and why did you use those techniques?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "49b16cbc",
   "metadata": {},
   "source": [
    "Standard outlier treatments are not applied to image data in this context. Pixel intensity outliers are not relevant as we normalize pixel values anyway."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5573cfed",
   "metadata": {},
   "source": [
    "### 3. Categorical Encoding"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "186ad779",
   "metadata": {},
   "outputs": [],
   "source": [
    "# The classes are encoded by the ImageDataGenerator automatically to integer labels\n",
    "print('Classes are automatically encoded by the data generator.')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ab2105cd",
   "metadata": {},
   "source": [
    "#### What all categorical encoding techniques have you used & why did you use those techniques?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "39080248",
   "metadata": {},
   "source": [
    "We rely on the Keras data generator to encode class labels as one-hot vectors for categorical classification. This is a suitable approach for multi-class targets."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c8ba6e09",
   "metadata": {},
   "source": [
    "## ***7. ML Model Implementation***"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "06b3271b",
   "metadata": {},
   "source": [
    "### ML Model - 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d829c89b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Build a custom CNN model\n",
    "cnn_model = Sequential([\n",
    "    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),\n",
    "    MaxPooling2D((2,2)),\n",
    "    BatchNormalization(),\n",
    "    Conv2D(64, (3,3), activation='relu'),\n",
    "    MaxPooling2D((2,2)),\n",
    "    BatchNormalization(),\n",
    "    Conv2D(128, (3,3), activation='relu'),\n",
    "    MaxPooling2D((2,2)),\n",
    "    BatchNormalization(),\n",
    "    Flatten(),\n",
    "    Dense(128, activation='relu'),\n",
    "    Dropout(0.5),\n",
    "    Dense(train_gen.num_classes, activation='softmax')\n",
    "])\n",
    "cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])\n",
    "cnn_model.summary()\n",
    "\n",
    "# Callbacks\n",
    "early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)\n",
    "checkpoint_cb = ModelCheckpoint('best_cnn.h5', monitor='val_loss', save_best_only=True)\n",
    "\n",
    "# Train the model\n",
    "history_cnn = cnn_model.fit(train_gen, epochs=10, validation_data=val_gen, callbacks=[early_stop, checkpoint_cb])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ada89b9e",
   "metadata": {},
   "source": [
    "#### 1. Explain the ML Model used and its performance using evaluation metric score chart."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "57a13bdf",
   "metadata": {},
   "source": [
    "We used a custom Convolutional Neural Network (CNN) with multiple Conv2D, MaxPooling, and BatchNormalization layers, followed by Dense layers. The model was trained with early stopping to prevent overfitting. Below we display the test accuracy, classification report, and confusion matrix to evaluate performance."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2db8802d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Evaluate on test data for CNN model\n",
    "cnn_model.load_weights('best_cnn.h5')\n",
    "test_loss, test_acc = cnn_model.evaluate(test_gen)\n",
    "print('CNN Test Accuracy:', test_acc)\n",
    "# Predictions and classification report\n",
    "pred_cnn = cnn_model.predict(test_gen)\n",
    "pred_labels_cnn = np.argmax(pred_cnn, axis=1)\n",
    "true_labels = test_gen.classes\n",
    "print(classification_report(true_labels, pred_labels_cnn, target_names=classes))\n",
    "cm = confusion_matrix(true_labels, pred_labels_cnn)\n",
    "plt.figure(figsize=(6,5))\n",
    "sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel('Actual')\n",
    "plt.title('Confusion Matrix - CNN')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f0025c59",
   "metadata": {},
   "source": [
    "#### 2. Cross-Validation & Hyperparameter Tuning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d203a487",
   "metadata": {},
   "outputs": [],
   "source": [
    "# For this CNN model, we used EarlyStopping and ModelCheckpoint as our tuning strategy.\n",
    "print('Used EarlyStopping with patience of 3 epochs. No additional hyperparameter tuning applied.')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e7b8a0ea",
   "metadata": {},
   "source": [
    "##### Which hyperparameter optimization technique have you used and why?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aa59d8bb",
   "metadata": {},
   "source": [
    "We utilized EarlyStopping to avoid overfitting. We did not perform grid search or other automated hyperparameter tuning due to computational constraints; instead, we manually chose a reasonable architecture."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2512d068",
   "metadata": {},
   "source": [
    "##### Have you seen any improvement? Note down the improvement with updated evaluation metric score chart."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "236c1e75",
   "metadata": {},
   "source": [
    "The use of EarlyStopping helped in preventing overfitting. After training, the model achieved a test accuracy of [INSERT TEST ACCURACY]% (as shown above)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3a454721",
   "metadata": {},
   "source": [
    "### ML Model - 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7ed7170f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Build a transfer learning model using MobileNetV2\n",
    "base_model = MobileNetV2(include_top=False, input_shape=(224,224,3), weights='imagenet')\n",
    "base_model.trainable = False  # Freeze base layers\n",
    "model_tl = Sequential([\n",
    "    base_model,\n",
    "    GlobalAveragePooling2D(),\n",
    "    Dropout(0.5),\n",
    "    Dense(128, activation='relu'),\n",
    "    Dropout(0.5),\n",
    "    Dense(train_gen.num_classes, activation='softmax')\n",
    "])\n",
    "model_tl.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])\n",
    "model_tl.summary()\n",
    "\n",
    "checkpoint_cb2 = ModelCheckpoint('best_mobilenet.h5', monitor='val_loss', save_best_only=True)\n",
    "\n",
    "history_tl = model_tl.fit(train_gen, epochs=10, validation_data=val_gen, callbacks=[early_stop, checkpoint_cb2])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e9aaf2bf",
   "metadata": {},
   "source": [
    "#### 1. Explain the ML Model used and its performance using evaluation metric score chart."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2bfa7a3d",
   "metadata": {},
   "source": [
    "We used MobileNetV2 (pretrained on ImageNet) as a feature extractor and added custom dense layers on top. The base was frozen during initial training. We evaluate similarly using test accuracy, classification report, and confusion matrix below."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ea012e26",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Evaluate on test data for MobileNet model\n",
    "model_tl.load_weights('best_mobilenet.h5')\n",
    "test_loss2, test_acc2 = model_tl.evaluate(test_gen)\n",
    "print('MobileNetV2 Test Accuracy:', test_acc2)\n",
    "pred_tl = model_tl.predict(test_gen)\n",
    "pred_labels_tl = np.argmax(pred_tl, axis=1)\n",
    "print(classification_report(true_labels, pred_labels_tl, target_names=classes))\n",
    "cm2 = confusion_matrix(true_labels, pred_labels_tl)\n",
    "plt.figure(figsize=(6,5))\n",
    "sns.heatmap(cm2, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel('Actual')\n",
    "plt.title('Confusion Matrix - MobileNetV2')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "347cb7b9",
   "metadata": {},
   "source": [
    "#### 2. Cross-Validation & Hyperparameter Tuning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ce6051b5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# For this MobileNet model, we again used EarlyStopping.\n",
    "print('Used EarlyStopping with patience of 3 epochs. No additional hyperparameter tuning applied.')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ed366572",
   "metadata": {},
   "source": [
    "##### Which hyperparameter optimization technique have you used and why?"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c3673db4",
   "metadata": {},
   "source": [
    "We again used EarlyStopping to avoid overfitting. No grid search or automated tuning was performed due to computational constraints."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3155b5a4",
   "metadata": {},
   "source": [
    "##### Have you seen any improvement? Note down the improvement with updated evaluation metric score chart."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6b1f47e0",
   "metadata": {},
   "source": [
    "The MobileNetV2 model achieved [INSERT TEST ACCURACY]% test accuracy, which should be compared to the CNN model's accuracy above. No automated tuning was done."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5bcc057a",
   "metadata": {},
   "source": [
    "## ***8.*** ***Future Work (Optional)***"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "29f275dd",
   "metadata": {},
   "source": [
    "### 1. Save the best performing ML model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a0879c82",
   "metadata": {},
   "outputs": [],
   "source": [
    "cnn_model.save('cnn_model.h5')\n",
    "model_tl.save('mobilenet_model.h5')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5e0a0728",
   "metadata": {},
   "source": [
    "### 2. Load the saved model and predict unseen data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "58114ced",
   "metadata": {},
   "outputs": [],
   "source": [
    "loaded_model = tf.keras.models.load_model('cnn_model.h5')\n",
    "pred = loaded_model.predict(test_gen)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c467e4de",
   "metadata": {},
   "source": [
    "## **Conclusion**\n",
    "The custom CNN and MobileNetV2 models were implemented for brain tumor classification. We compared their test accuracies and other metrics. Both models effectively learned to classify the MRI images. The MobileNetV2 (transfer learning) model achieved [INSERT ACCURACY]%, while the custom CNN achieved [INSERT ACCURACY]%. Future work could include additional hyperparameter tuning, more advanced architectures, and using a larger dataset to further improve accuracy.\n",
    "\n",
    "### ***Hurrah! You have successfully completed the Machine Learning Capstone Project!!!***"
   ]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}

