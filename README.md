# Spikemax Loss and suppressed loss for Spiking neural network
**What is this repository for?**

spike-based negative log-likelihood-based losses suitable to train an SNN for classification tasks.

**Usage**

This loss can be easily implemented by replacing this with the loss layer. You can also directly implement it into SLayer framework 
 https://github.com/bamsumit/slayerPytorch. The Spikemax loss code can also be found at https://github.com/lava-nc/lava-dl/blob/main/src/lava/lib/dl/slayer/loss.py.

error = spikeLoss(netParams).to(device)

loss = error.spikeRate(output, target)

loss_supressed =  alpha * error.loss_mem(output, target, mem)

loss.backward()

In this context, "output" refers to the network's output, "target" represents the network's label, and "mem" corresponds to the membrane potential of the final layer. The parameter "alpha" is set to its default value of 0.1.

**Papers**

Spikemax loss

@inproceedings{shrestha2022spikemax,
  title={Spikemax: Spike-based loss methods for classification},
  author={Shrestha, Sumit Bam and Zhu, Longwei and Sun, Pengfei},
  booktitle={2022 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--7},
  year={2022},
  organization={IEEE}
}


Suppressed loss

@article{sun2023learnable,
  title={Learnable axonal delay in spiking neural networks improves spoken word recognition},
  author={Sun, Pengfei and Chua, Yansong and Devos, Paul and Botteldooren, Dick},
  journal={Frontiers in Neuroscience},
  volume={17},
  year={2023},
  publisher={Frontiers Media SA}}
