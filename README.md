# End-to-End-Pipeline-for-Robotics-ML-Training_NEW_OBJECT_EXTERNAL
End to End pipeline to use Nvidia's TAO Toolkit, based on Nvidia's tutorial, and adapts it to train a model to find any object of interest (in this example, cardboxes)

# End-to-End-Pipeline-for-Robotics-ML-Training-with-Synthetic-Data-on-Nvidia-Isaac-SIM

## Describing the different files used in the process:

- generate_data.sh: 
  - Path:"C:[REPLACE WITH YOUR LOCAL PATH]GitHub\synthetic_data_generation_training_workflow\local\generate_data.sh"
  - NEW one for cardboard boxes: "generate_data_cardbox_GPT" (for Windows)
  - Path: "C:[REPLACE WITH YOUR LOCAL PATH]GitHub\synthetic_data_generation_training_workflow\local\generate_data_cardbox_GPT.sh"
  - What is it?: This script calls the data generation script "standalone_palletjack_sdg.py" (standalone_cardbox_sdg_GPT.py for the cardbox generation) and passes parameters to it. Because we want to create 3 different types of distractor objects, we need to run the data generation script 3 times, each time, passing different parameters. This scrip automates the process of running the data generation script 3 times. 
  - It saves the synthetic images generated in the folders saved in the "OUTPUT_..." variables.

- standalone_palletjack_sdg.py
  - Path: "C:[REPLACE WITH YOUR LOCAL PATH]GitHub\synthetic_data_generation_training_workflow\palletjack_sdg\standalone_palletjack_sdg.py"
  - NEW one for cardboard boxes: "standalone_cardbox_sdg_GPT"
  - Path: "C:[REPLACE WITH YOUR LOCAL PATH]GitHub\synthetic_data_generation_training_workflow\cardbox_sdg\standalone_cardbox_sdg_GPT.py"
  - What is it?: This is the script that actually generates the synthetic data (images). 
  - What it does:?
   - Opens a warehouse scene in NVIDIA Isaac Sim.
   - Spawns a few pallet jack assets (the object we want to detect).
   - Spawns distractor props (extra objects) to make scenes varied.
   - Randomizes camera, lighting, object poses, and materials (domain randomization).
   - Captures images + labels using a KITTI-format "writer" to build a training dataset.
   - In a nutshell, It loads a warehouse 3D environment from Nvidia, fetches 3D assets from the Nvidia cloud database (both objects of interest and distractors), then uses the Replicator function to randomize the position and color of these objects in the scene. 

- local_train.ipynb
  - Path: "C:[REPLACE WITH YOUR LOCAL PATH]GitHub\synthetic_data_generation_training_workflow\local\local_train.ipynb"
  - NEW one for cardboard boxes: "local_train_cardbox_GPT"
  - Path: "C:[REPLACE WITH YOUR LOCAL PATH]GitHub\synthetic_data_generation_training_workflow\local\local_train_cardbox_GPT.ipynb"
  - What is it?: A Python script on Jupyter Notebook that takes the synthetic images, trains a ML model on them and tests it to try to find the object of interest and then measures its precision. 


## STEP BY STEP

### 1. Generate Synthetic Data

#### 1.1. Create New Folders (For Cardboxes)
- "cardbox_sdg" folder: 
  - Create it here: Path: "C:[REPLACE WITH YOUR LOCAL PATH]GitHub\synthetic_data_generation_training_workflow\cardbox_sdg"
  - Inside "cardbox_sdg" folder: 
  - Put "standalone_cardbox_sdg_GPT" inside
  - create "cardbox_data" folder
   - Create it here: Path: "C:[REPLACE WITH YOUR LOCAL PATH]GitHub\synthetic_data_generation_training_workflow\cardbox_sdg\cardbox_data"
   - Inside "cardbox_data" folder
   -  create "distractors_warehouse" folder
   - Create "distractors_additional" folder
   - Create "no_distractors" folder
-  "resnet18_cardbox" folder:
  - Create it here: Path: "C:[REPLACE WITH YOUR LOCAL PATH]GitHub\synthetic_data_generation_training_workflow\local\training\tao\detectnet_v2\resnet18_cardbox"
  - This is where the ML model will save the images with annotations (identification boxes) and weights.

From COURSE: Synthetic Data Generation for Perception Model Training in Isaac Sim
https://learn.nvidia.com/courses/course?course_id=course-v1:DLI+S-OV-30+V1&unit=block-v1:DLI+S-OV-30+V1+type@vertical+block@7fecaf9f66204c0ea35402fca5ae1b25
Generating a Synthetic Dataset Using Replicator > Activity: Understanding Basics of the SDG Script and Activity: Running the Script for Generating Training Data

#### 1.2. Clone repo: 
- git clone https://github.com/NVIDIA-AI-IOT/synthetic_data_generation_training_workflow.git
- OK: Cloned to "C:[REPLACE WITH YOUR LOCAL PATH]GitHub"

#### 1.3. Adjust generate_data.sh to clear cache
- NEW file for cardboxes: "generate_data_cardbox_GPT"
- Open file with text editor
- add "--clear-cache --clear-data" to the end of the parameters list passed for each run. eg: "cmd //c "C:\isaacsim\python.bat" $SCRIPT_PATH --height 544 --width 960 --num_frames 2000 --distractors warehouse --data_dir $OUTPUT_WAREHOUSE --clear-cache --clear-data"
. This is to make sure after each run the cache is deleted to prevent the GPU from running out of memory

#### 1.4. Manually Delete Cache Files
- If previous runs have been done before, delete cache files from "C:\Users\myali\AppData\Local\ov\cache"
- This is to make sure after each run the cache is deleted to prevent the GPU from running out of memory

#### 1.5. Run generate_data.sh
- 1.3.1. Locate and Configure generate_data.sh
 - 2.1- Go to my address: "C:[REPLACE WITH YOUR LOCAL PATH]GitHub\synthetic_data_generation_training_workflow\local". Open the file "generate_data.sh" with a text editor.
  - 2.2- Open file and insert the path where I saved Isaac SIM on my computer: eg: "C:\isaacsim"
  - 2.3. Check the path assigned to the "output variables", this is where the images will be saved

1.3.2. Run the script
- Go to the folder and double click on the file: "generate_data.sh"
   - NEW one: "generate_data_cardbox_GPT.sh"
- it will open Isaac SIM and start generating synthetic data image files to the output folders

#### 1.6. Check synthetic data generated
- Go to path assigned to the output variables 
- Path: "C:[REPLACE WITH YOUR LOCAL PATH]GitHub\synthetic_data_generation_training_workflow\cardbox_sdg\carbox_data"

_____________________________________________________________________________


### 2: Train and Test the Machine Learning Model with the generated Synthetic Data 

From COURSE: "Fine-Tuning and Validating an AI Perception Model" > "Lecture: Training a Model With Synthetic Data"
https://learn.nvidia.com/courses/course?course_id=course-v1:DLI+S-OV-30+V1&unit=block-v1:DLI+S-OV-30+V1+type@vertical+block@aced7cf26b974581baf48fae53b70341

#### 2.1: Training Environment Setup

- 0 - Make sure you have completed all the previous steps to generate synthetic data: 
  - Clone repo: https://github.com/NVIDIA-AI-IOT/synthetic_data_generation_training_workflow.git
  - Configure generate_data.sh
  - Run generate_data.sh

- 1- Install Ubuntu and Open Ubuntu CLI
  - Some of the code in the jupyter notebook that runs the model is for Linux, so if you are running from a windows machine you need to install Ubuntu (Linux environment for Windows) and run everything from the Ubuntu CLI 

- 2- create a separate Conda environment that users python version 3.10: 
  - "conda create -n tao-py310 python=3.10 
  - If this environment has already been previously created, skip this step.

- 3- Activate python 3.10 conda env: 
  - "conda activate tao-py310"

- 4- Connect to Nvidia's docker container:
  - The jupyter notebook that runs this model needs to be run from a Docker container that has all libraries already installed
  - Open docker desktop application and click on the play button on the container in the container list
  - in the ubuntu cli run "docker login nvcr.io"
  - login to the Nvidia container (if already logged user and password were already saved, if not get user (API key) from https://org.ngc.nvidia.com/setup/api-keys and rotate password to get a new password)
  - run "docker ps -a" to check active containers.

- 5- Navigate to the mounted folder in Ubuntu with the jupyter notebook
  - navigate, in Ubuntu CLI, to folder where the GitHub project for synthetic data generation was cloned (from project https://learn.nvidia.com/courses/course?course_id=course-v1:DLI+S-OV-30+V1&unit=block-v1:DLI+S-OV-30+V1+type@vertical+block@7fecaf9f66204c0ea35402fca5ae1b25: "Generating a Synthetic Dataset Using Replicator > Activity: Understanding Basics of the SDG Script": 
"cd /mnt/c/Users/myali/Documents/GitHub/synthetic_data_generation_training_workflow/local"

6- Open the notebook in this folder from Ubuntu CLI: 
  - "jupyter notebook local_train.ipynb --allow-root"
  - NEW one: local_train_cardbox_GPT
  - Copy the URL provided in my web browser, click on the notebook to open


#### 2.2: Adjust spec files

##### Edit Tfrecords spec files : "distractors_additional, distractors_warehouse, no_distractors"

- Change: "distractors_additional, distractors_warehouse, no_distractors" Kitti config files
- Path: "C:[REPLACE WITH YOUR LOCAL PATH]GitHub\synthetic_data_generation_training_workflow\local\training\tao\specs\tfrecords"
- duplicate: "distractors_additional, distractors_warehouse, no_distractors" > create new "distractors_additional_cardbox, distractors_warehouse_cardbox, no_distractors_cardbox"
- replace: inside it:
   . all "palletjack" for "cardbox"
   . Add, indented under "image_directory_path:":
      "target_class_mapping {
         key: "cardbox"
         value: "cardbox"
      }"
   . Example:
"image_directory_path: "/workspace/tao-experiments/cardbox_sdg/cardbox_data/no_distractors/Camera"
  target_class_mapping {
      key: "cardbox"
      value: "cardbox"
  }"

##### Edit Training spec file: "resnet18_distractors"

- Change: "resnet18_distractors" config file
Path: "C:[REPLACE WITH YOUR LOCAL PATH]GitHub\synthetic_data_generation_training_workflow\local\training\tao\specs\training"
- duplicate: "resnet18_distractors" > create new "resnet18_distractors_cardbox"
- replace: inside it, all "palletjack" for "cardbox"
- In: 
   - dataset_config { data_sources
   - target_class_mapping
   - postprocessing_config
   - evaluation_config
   - cost_function_config 
   - visualizer
   - bbox_rasterizer_config

##### Edit Inference spec file: "new_inference_specs"

- Change: "new_inference_specs" config file
Path: "C:[REPLACE WITH YOUR LOCAL PATH]GitHub\synthetic_data_generation_training_workflow\local\training\tao\specs\inference"
- duplicate: "new_inference_specs" > create new "new_inference_specs_cardbox"
- replace: inside it, all "palletjack" for "cardbox"



#### 2.2: Adjust code in the jupyter notebook

- Adjust code in the local_train.ipynb file

- inside the notebook: DO NOT replace "# os.environ["LOCAL_PROJECT_DIR"] = "<LOCAL_PATH_OF_CLONED_REPO>" this line of code doesn't do anything since the path is automatically fetched. 

##### "3. Convert Dataset to TFRecords for TAO"

###### Data Cleanup 1: Filter out problematic image and label files (not in the original documentation)

- The code and models provided in the original documentation are brittle. They will eventually produce label files with size = 0, blank images, bounding boxes with width = 0, tfrecord files with size = 0 etc. 
- These data cleanup scripts are meant to correct that and delete such files 
- Before "3. Convert Dataset to TFRecords for TAO", add this script: 
```
import os
import logging
from PIL import Image
import numpy as np

# --- GENERAL-PURPOSE DATA VALIDATION & CLEANUP SCRIPT ---
# This script automatically discovers and validates all dataset subfolders.
# It checks for and deletes pairs with:
# 1. Missing or empty label files.
# 2. Corrupt or blank (single-color) image files.
# 3. Invalid (zero-width or zero-height) bounding boxes inside label files.

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- CONFIGURATION ---
# The script will start from the project root and build the path from there.
project_root = os.getenv('LOCAL_PROJECT_DIR', '/workspace/tao-experiments')

# This is the main folder that contains all your dataset subfolders.
all_datasets_base_path = os.path.join(project_root, "cardbox_sdg", "cardbox_data")
# --------------------

logging.info(f"--- Starting General Validation for all datasets in: {all_datasets_base_path} ---")

if not os.path.isdir(all_datasets_base_path):
    logging.error(f"Base dataset directory not found: {all_datasets_base_path}")
else:
    # Automatically find all subdirectories in the base path
    datasets_to_check = [d for d in os.listdir(all_datasets_base_path) if os.path.isdir(os.path.join(all_datasets_base_path, d))]

    if not datasets_to_check:
        logging.warning("No dataset subfolders found to validate.")
    else:
        logging.info(f"Found datasets to validate: {datasets_to_check}")
        
        total_invalid_pairs = 0

        # Loop through each discovered dataset
        for dataset_name in datasets_to_check:
            dataset_path = os.path.join(all_datasets_base_path, dataset_name, "Camera")
            images_dir = os.path.join(dataset_path, "rgb")
            labels_dir = os.path.join(dataset_path, "object_detection")
            invalid_pairs_in_set = 0

            logging.info(f"\n--- Validating Dataset: {dataset_name} ---")

            if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
                logging.warning(f"Skipping '{dataset_name}' because 'rgb' or 'object_detection' subfolder is missing.")
                continue

            image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
            
            for image_name in image_files:
                image_path = os.path.join(images_dir, image_name)
                label_name = os.path.splitext(image_name)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_name)
                
                delete_pair = False
                reason = ""

                # Check 1: Label file is missing or empty
                if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
                    delete_pair = True
                    reason = "Label file is missing or empty."
                
                # Check 2: Image file is blank or corrupt
                if not delete_pair:
                    try:
                        with Image.open(image_path) as img:
                            img.load()
                            if np.std(np.array(img)) < 1:
                                delete_pair = True
                                reason = "Image file is blank (single color)."
                    except Exception as e:
                        delete_pair = True
                        reason = f"Image file is corrupt or unreadable ({e})."

                # Check 3: Bounding boxes in label are invalid
                if not delete_pair:
                    try:
                        with open(label_path, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) == 15:
                                    left, top, right, bottom = map(float, parts[4:8])
                                    if left >= right or top >= bottom:
                                        delete_pair = True
                                        reason = f"Label contains invalid bounding box (left={left}, right={right})."
                                        break
                    except Exception as e:
                        delete_pair = True
                        reason = f"Could not parse label file ({e})."

                # If any check failed, delete the image/label pair
                if delete_pair:
                    invalid_pairs_in_set += 1
                    logging.warning(f"DELETING: '{image_name}' and its label. Reason: {reason}")
                    if os.path.exists(image_path): os.remove(image_path)
                    if os.path.exists(label_path): os.remove(label_path)

            logging.info(f"--- Validation for '{dataset_name}' complete. Found and deleted {invalid_pairs_in_set} invalid pairs. ---")
            total_invalid_pairs += invalid_pairs_in_set

        print(f"\n--- TOTAL CLEANUP COMPLETE: Deleted {total_invalid_pairs} total invalid pairs from all datasets. ---")

```
###### Adapt dataset_convert code

  - replace:
```
!mkdir -p $LOCAL_PROJECT_DIR/local/training/tao/tfrecords/distractors_warehouse && rm -rf $LOCAL_PROJECT_DIR/local/training/tao/tfrecords/distractors_warehouse/*

   !docker run -it --rm --gpus all -v $LOCAL_PROJECT_DIR:/workspace/tao-experiments $DOCKER_CONTAINER \
                   detectnet_v2 dataset_convert \
                  -d /workspace/tao-experiments/local/training/tao/specs/tfrecords/distractors_warehouse.txt \
                  -o /workspace/tao-experiments/local/training/tao/tfrecords/distractors_warehouse/
```
  - with:
```
!mkdir -p $LOCAL_PROJECT_DIR/local/training/tao/tfrecords/distractors_warehouse && rm -rf $LOCAL_PROJECT_DIR/local/training/tao/tfrecords/distractors_warehouse/*

   !docker run -it --rm --gpus all -v $LOCAL_PROJECT_DIR:/workspace/tao-experiments $DOCKER_CONTAINER \
                  detectnet_v2 dataset_convert \
                  -d /workspace/tao-experiments/local/training/tao/specs/tfrecords/distractors_warehouse_cardbox.txt \
                  -o /workspace/tao-experiments/local/training/tao/tfrecords/distractors_warehouse/"

  - Do the same for the other kitti conversion code blocks. For: "distractors_additional" and "no_distractors
  ```

-> dataset_convert documentation: https://docs.nvidia.com/tao/tao-toolkit/text/cv_finetuning/tensorflow_1/object_detection/detectnet_v2.html#id2

###### Data Cleanup 2: Remove 0kb size Tfrecord files (not in the original documentation)

- after running dataset_convert "3. Convert Dataset to TFRecords for TAO ", insert this code snipt to delete tfrecord files with size = 0:
```
# This command finds all files (-type f) of size 0 (-size 0) in the tfrecords directory and deletes them (-delete). It also prints the names of the files it deletes
# (this wasn't in the original documentation)
print("Searching for and deleting any 0kb TFRecord files...")
!find $LOCAL_PROJECT_DIR/local/training/tao/tfrecords -type f -size 0 -delete -print
print("Deletion of 0kb files complete.")

```

##### "4. Provide Training Specification File"

- edit: the code in the notebook. replace: "!cat $LOCAL_PROJECT_DIR/local/training/tao/specs/training/resnet18_distractors.txt" by "!cat $LOCAL_PROJECT_DIR/local/training/tao/specs/training/resnet18_distractors_cardbox.txt"

 - 4.1: Delete Tfrecord files with 0KB size (not in the original documentation)
 . detectnet_v2 dataset_convert converts input images into Tfrecord files that are easier to be read by tensorflow.
 . if detectnet_v2 dataset_convert doesn't find the object of interest in the source image file, it creates an empty output: a 0kb size Tfrecord file. 
 . During training, if detectnet_v2 train finds a 0kb file, it errors throwing "tensorflow.python.framework.errors_impl.DataLossError: corrupted record at 0"
 . to avoid this from happening, add the following code snipet before step 5. Training:
 ```
 # This command finds all files (-type f) of size 0 (-size 0) in the tfrecords directory and deletes them (-delete). It also prints the names of the files it deletes
# (this wasn't in the original documentation)
print("Searching for and deleting any 0kb TFRecord files...")
!find $LOCAL_PROJECT_DIR/local/training/tao/tfrecords -type f -size 0 -delete -print
print("Deletion of 0kb files complete.")
```

##### "5. Run TAO Training"

- edit: the code in the notebook. replace: "-e /workspace/tao-experiments/local/training/tao/specs/training/resnet18_distractors.txt" by "-e /workspace/tao-experiments/local/training/tao/specs/training/resnet18_distractors_cardbox.txt"

- in the notebook, replace "!rm -rf $LOCAL_PROJECT_DIR/local/training/tao/detectnet_v2/resnet18/*" by "!rm -rf $LOCAL_PROJECT_DIR/local/training/tao/detectnet_v2/resnet18_cardbox/*"

##### "6. Evaluate Trained Model"

- edit: the code in the notebook. replace: "-e /workspace/tao-experiments/local/training/tao/specs/training/resnet18_distractors.txt" by "-e /workspace/tao-experiments/local/training/tao/specs/training/resnet18_distractors_cardbox.txt" 

##### "7. Visualize Model Performance on Real World Data"

- edit: the code in the notebook. replace: "-e /workspace/tao-experiments/local/training/tao/specs/inference/new_inference_specs.txt" by "-e /workspace/tao-experiments/local/training/tao/specs/inference/new_inference_specs_cardbox.txt"

#### 2.4 run all cells in the notebook


