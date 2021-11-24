# Using DeepSpeed with Determined
DeepSpeed is a library released by Microsoft that supports large-scale distributed learning with
sharded optimizer state training and pipeline parallelism.  Using DeepSpeed with Determined is 
supported through our `DeepSpeedTrial` API.  

DeepSpeed Features:
* Zero Redundancy Optimizer (ZeRO) with parameter and optimizer state offloading
* Pipeline parallelism with interleaved microbatch training

Known Limitations of DeepSpeed:
* The primary limitation to be aware of is that pipeline parallelism can only be combined with ZeRO stage 1.  
* Parameter offloading is only supported with ZeRO stage 3.
* Optimizer offloading is only supported with ZeRO stage 2 and 3.

## Basics of DeepSpeed

### Configuration
DeepSpeed is usually used with a configuration file specifying the settings for various DeepSpeed features.
In lieu of Determined, this configuration file is passed when launching a training job:
```
deepspeed train.py --deepspeed_config=ds_config.json 
```
The configuration file path is parsed into an arguments object (e.g. `args`), which is used later to 
create the DeepSpeed model engine.

### Initialization
Initializing DeepSpeed training consists of two steps:
1. Initializing the distributed backend.
2. Creating the DeepSpeed model engine.

This is usually done with something like below:
```python
import deepspeed

...

deepspeed.init_distributed(dist_backend=args.backend) # backend usually nccl
net = ...
model_engine, optimizer, lr_scheduler, dataloader = deepspeed.initialize(
    args=args, # args has deepspeed_config as an attribute
    net=net,
    ...
)
```

### Training
Once the DeepSpeed model engine is created, the training process will depend on whether pipeline
parallelism is being used.

#### Data Parallel Training
For just data parallel training, the forward and backward steps are performed as follows:
```python
outputs = model_engine(inputs)
loss = criterion(outputs, targets)
model_engine.backward(loss)
model_engine.step()
```
Note how `backward` and `step` are called on the DeepSpeed model engine instead of `loss` and the optimizer respectively.

#### Pipeline Parallel Training
If a user wants to use pipeline parallelism, they will need to pass layers of their model to 
DeepSpeed's PipelineModule before creating the DeepSpeed model engine:
```python
net = PipelineModule(
    layers=get_layers(net), # get_layers is a user provided function that will return layers of a network.
    loss_fn=torch.nn.CrossEntropyLoss(),
    num_stages=args.pipeline_parallel_size,
    ...
)
model_engine, _, _, _ = deepspeed.initialize(
    args=args,
    model=net,
    dataset=dataset, # optional
    ...
)
```
When using pipeline parallelism, DeepSpeed expects the configuration file to have `train_batch_size` and
`train_micro_batch_size_per_gpu` to be available so it can automatically interleave multiple microbatches
for processing in a single training schedule.  
If a `dataset` is passed to `deepspeed.initialize`, the model_engine will build an internal data loader that
creates batches of size `train_micro_batch_size_per_gpu`, which can be specified in the DeepSpeed config. 
You can also create your own dataloader and use that directly.
```python
for _ in range(train_iters):
    # The model_engine will automatically perform forward, backward, and optimizer update on 
    # batches requested internally from the dataloader and interleave.  
    model_engine.train_batch(dataloader) 
```

### Putting it together

#### Data Parallel Training
For just data parallel training, the forward and backward steps are performed as follows:
```python
deepspeed.init_distributed(dist_backend=args.backend) # backend usually nccl
net = ...
model_engine, optimizer, lr_scheduler, dataloader = deepspeed.initialize(
args=args, # args has deepspeed_config as an attribute
net=net,
...
)
train_dataloader = ...
for idx, batch in enumerate(dataloader):
    inputs, targets = batch
    outputs = model_engine(inputs)
    loss = criterion(outputs, targets)
    model_engine.backward(loss)
    model_engine.step()
```

#### Pipeline Parallel Training
```python
net = PipelineModule(
    layers=get_layers(net), # get_layers is a user provided function that will return layers of a network.
    loss_fn=torch.nn.CrossEntropyLoss(),
    num_stages=args.pipeline_parallel_size,
    ...
)
model_engine, _, _, _ = deepspeed.initialize(
    args=args,
    model=net,
    dataset=dataset, # optional
    ...
)
train_dataloader = ...
for _ in range(train_iters):
    # The model_engine will automatically perform forward, backward, and optimizer update on 
    # batches requested internally from the dataloader and interleave.  
    model_engine.train_batch(train_dataloader) 
```

## Using DeepSpeedTrial
You can think of `DeepSpeedTrial` as a way to use an automated training loop with DeepSpeed.  
Next, we'll demonstrate how the typical usage of DeepSpeed maps over to Determined.

### Determined's Experiment Configuration
Configuration Determined experiments for DeepSpeed is largely the same as doing so for PyTorchTrial 
with a few differences. 
* You will need to specify a required `hyperparameter` called `data_parallel_world_size` to explicitly
tell Determined how many model replicas you expect there to be.  
* You still need to provide `hyperparameters.global_batch_size` but you should make sure that this
matches `train_batch_size` in the DeepSpeed config.

You have control over how you pass a DeepSpeed configuration file for use to initialize the DeepSpeed model engine.  
One natural way is to specify it as a hyperparameter and treat the hyperparameters field as arguments
to pass to DeepSpeed.  

Your Determined experiment config might look something like this:
```yaml
hyperparameters:
  global_batch_size: 32 # this should match the train_batch_size you set in your DeepSpeed config
  data_parallel_world_size: 2
  deepspeed_config: ds_config.json
  ...
```

### Implementing the Trial API

#### Initializing 
```diff
import deepspeed
from determined.pytorch import DataLoader, DeepSpeedTrial, DeepSpeedTrialContext, DeepSpeedMPU

class MyTrial(DeepSpeedTrial):
    def __init__(self, context: DeepSpeedTrialContext) -> None:
        self.context = context
        self.args = AttrDict(self.context.get_hparams())
        model = MyModel(...)
        self.model_engine, _, _, _ = deepspeed.initialize(
            args=self.args, 
            model=model, 
            ...
        )
+        self.mpu = self.context.wrap_mpu(DeepSpeedMPU(model_engine.mpu))
+        self.model_engine = self.context.wrap_model_engine(model_engine)

    def build_training_data_loader(self) -> Any:
        trainset = ...
        return DataLoader(trainset, batch_size=self.args.train_micro_batch_size_per_gpu, shuffle=True)

    def build_validation_data_loader(self) -> Any:
        valset = ...
        return DataLoader(valset, batch_size=self.args.train_micro_batch_size_per_gpu, shuffle=False)

    def train_batch(
        self, iter_dataloader: Iterable[DataLoader], epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        input, target = next(iter_dataloader)
        outputs = self.model_engine(
        loss = self.model_engine.train_batch(iter_dataloader)
        return {"loss": float(loss)}

    def evaluate_batch(self, iter_dataloader: Iterable[DataLoader]) -> Dict[str, Any]:
        loss = self.model_engine.eval_batch(iter_dataloader)
        return {"loss": loss}
```


## Advanced Usage
### Checking DeepSpeed & Determined arguments compatibility
deepspeed.DeepSpeedConfig(args.deepspeed_config,)

self.context.reconcile_ds_config(model_engine._config)
### Overwriting DeepSpeed config arguments 
self.context.overwrite(deepspeed)
