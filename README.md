Code and models for [Coupled Ensembles of Neural Networks](http://arxiv.org/abs/1709.06053) 

  @article{dutt2017coupledEnsembles,
    title={Coupled Ensembles of Neural Networks},
    author={Dutt, Anuvabh and Pellerin, Denis and Qu{\'}enot, Georges},
    jounral={ArXiv e-prints},
    eprint={1709.06053},
    year={2017},
  }


```
# The options for training can be specified in config.yaml.
# Refer to the file for an example. It shows how to train a 
# a couple ensemble with `E` branches, with a basic block 
# of DenseNet-BC depth=100, growthRate=12

python train_model.py --configFile config.yaml
```


The `models` folder contains the architecture definitions. To
experiment with other architectures, add a the model definition
in that folder.
