# CSAW-HACK-ML
ECE GY 9163 Final Project

Anup Upasani (asu224)

Neeraja Narayanswamy (nn2108)

Priyanka Shishodia (ps4118)


## Instructions

## Step 1: Downloading Data

Go into the data folder herein and copy the link in the Google Drive Link text file or get the link from below.

Download all data as is and put it into the data folder

Link to data files : https://drive.google.com/drive/folders/13o2ybRJ1BkGUvfmQEeZqDo1kskyFywab?usp=sharing

## Step 2: Checking Imports

The following imports are required. Please install them if they are not already installed

	keras, tensorflow, h5py, numpy, matplotlib, random, opencv-python, datetime, scipy, imageio
	
Note that sys, shutil, and math are required as well, but these should already be part of the python installation

## Step 2.5 (Optional): Checking Code

To see how the repair.py code generates test images, run the following command using a command prompt inside the eval folder herein
		
	cd eval/
	python3 repair.py
		
Note that this will take only a few minutes to run at maximum

To see how the repair.py code generates GoodNet models, run the following command using a command prompt inside the eval folder herein
		
	python3 repair.py init
		
Note that this will take 30+ minutes to run
		
To see how the repair.py code generates GoodNet models and sets up STRIP, run the following command using a command prompt inside the eval folder herein
	
	python3 repair.py init complex
		
Note that this will take 4+ hours to run
		
## Step 3: Evaluating H5 Files

Note: Please place any NEW h5 files to test inside the eval folder herein (NOT inside eval/poisoned_images).
	
To evaluate an h5 file (clean or poisoned) with any eval script, use the following syntax, where items in brackets are user inputs specified below. Run the command using a command prompt inside the eval folder herein
	
		python3 [eval_script] [h5_filename]
		
		Where,
			[eval_script] can be the following options:

				eval_sunglasses.py, eval_anonymous1.py, eval_anonymous2.py, eval_mtmt.py

				(Which corresponds to the sunglasses, anonymous 1, anonymous 2, and multi-target multi-trigger networks, respectively)

			[h5_filename] is the h5 filename (with extension) that was added to the eval folder
			

		For example,

			python3 eval_sunglasses.py data.h5
		

