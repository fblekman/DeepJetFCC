DeepJetFCC: From FCCAnalyses to classfied jets
===============================================================================

The prerequisite for classifying jets is a properly formatted ntuple. An example (with the necessary functions) in FCCAnalyses is given here: https://github.com/Edler1/FCCAnalyses-1/blob/DeepJetTransformer/examples/FCCee/TNtupler/analysis.py). 

This package depends on DeepJetCore 3.X.

The training/predicting pipeline pictured below will be briefly described
![alt text](https://github.com/Edler1/DeepJetFCC/blob/master/docs/pipeline.png)


Setup
==============

Begin by cloning this repo and DeepJetCore (https://github.com/AlexDeMoor/DeepJetCore) to some folder:
```
mkdir <your working dir>/DeepJetFCC && cd DeepJetFCC 
git clone https://github.com/AlexDeMoor/DeepJetCore
git clone https://github.com/Edler1/DeepJetFCC
```

Source the singularity container, and compile DeepJetCore:
```
source DeepJetFCC/sing_env.sh 
cd DeepJetCore 
source docker_env.sh
cd compiled && make -j8 
```

Now the packages should be setup and ready for use.

*The below scripts depend on numpy and pytorch which can be installed with* 
```
pip3 install torch
```
*if not already present.* 

Postprocessing 
--------------

Every new set of ntuples must be postprocessed once to reshape them from event/event to jet/jet. This can be done by invoking the postprocessor_2d.cpp. 

*N.B. The postprocessing script should be run in the base env of lxplus, not within the singularity container.* 

```
g++ postprocessor_2d.cpp -o postprocessor_2d.exe `root-config --cflags --glibs`
./postprocessor_2d.exe file_to_be_processed.root uds
```
For multiple files (and flavours), the postprocess.sh script could come in handy.

The output directory should list the files in .txt files to be used for training/testing (which should never overlap).

```
ls <your output directory>/*.root > <your output directory>/train.txt 
```


Usage
====

Source the singularity image
```
source sing_env.sh
```
Then setup the environment 
```
source setup.sh
```

Converting to .djctd
--------------------

Training and inference in the DeepJet framework requires that the .root files be converted to .djctd. This will save only the relevant information in the files and convert them to a format suitable for pytorch. This is done by running convertFromSource.py:

```
python3 ../DeepJetCore/bin/convertFromSource.py -i /path/to/postprocessed/root/ntuple/train.txt -o /output/path/that/needs/some/disk/space -c TrainData_ParT
``` 

Training  
--------------

In practise it is much faster to train (and infer) on GPUs. There are GPUs available on lxbatch that can be requested using the sing.sub script in lxplusGPUs. Their availability varies, but usually one should be assigned a GPU within a couple minutes. Since training can take several hours and a disconnect can kill the job, it is advisable (but not necessary) to use a screen session when training. This is done by:

```
cd lxplusGPUs
screen
``` 

*you will now be in the screen and can submit the job*

```
condor_submit -interactive sing.sub
```

*you should see a "Waiting for job to start..."*

Once the job starts it is necessary to swap to your home/working directory and source the aforementioned scripts

```
cd /afs/cern.ch/your/home_or_working/directory/DeepJetFCC/DeepJetFCC 
source sing_env.sh
source setup.sh
```  

At this point the network can be trained with the prepared files

```
python3 pytorch/train_DeepFlavour.py /path/to/the/converted/output/dataCollection.djcdc <output dir of your choice>
```

While the network trains (or at any other point) the screen can be deattached by using *ctrl-a d*. The screen can be reattached/resumed with *screen -r* on the same machine (e.g. lxplus725).

*Remember to end screen sessions (ctrl-a k or exit) when you are done using them. This is especially important when using a GPU.*

Predicting
-----------

To predict/infer using a trained model the predict_pytorch.py script is run. Some of the above steps (like requesting a GPU) will be helpful.

```
python3 pytorch/predict_pytorch.py <DeepJet/DeepJetTransformer/DeepJetTransformerV0> <output dir of training>/checkpoint.pth <output dir of training>/trainsamples.djcdc <dir with test sample stored as rootfiles>/filelist.txt <output directory> -b 4000
```

The output will be a .npz file and a .root file saved to the training directory.

Miscellaneous
=============

Changing training variables 
---------------------------

This is done by altering the datastructure found in DeepJetFCC/modules/datastructures/TrainData_deepFlavour.py. For instance adding a global feature would be done by appending to the list

```
self.global_branches = ['feature_1', ...] -> self.global_branches = ['feature_1', ..., 'added_feature']
```

This will require altering some params in the network file pytorch/pytorch_deepjet_transformer.py. The easiest way to find which is to use the errors output during training.

Changing the variables that are output during inference
-------------------------------------------------------

This is done by also altering the datastructure, but in particular the line 

```
self.spectator_branches = ['event_index', 'jets_px', 'jets_py', ...] -> self.spectator_branches = ['event_index', 'jets_px', 'jets_py', ..., 'added_feature']
```

In addition, it is necessary to alter the prediction script pytorch/predict_pytorch.py 

```
fields=["event_index", "jets_px", "jets_py", "predicted", "truths", ...]
```

to match spectator_branches **exactly**. If this is not done, the raw_predictions.root file will be nonsensical. 






