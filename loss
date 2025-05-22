import slayerSNN
import slayerCuda
import torch
import numpy as np

class spikeLoss(slayerSNN.spikeLoss.spikeLoss):
    def __init__(self, networkDescriptor, slayerClass=slayerSNN.layer):
        super(spikeLoss, self).__init__(networkDescriptor, slayerClass)
        self.slidingWindow = None
        
    def spikeRate(self, spikeOut, desiredClass, scale=1, earlySpikeBias=None):
        assert self.errorDescriptor['type'] == 'SpikeRate', "Error type is not SpiekRate"

        tgtSpikeRegion = self.errorDescriptor['tgtSpikeRegion']
        tgtSpikeRate   = self.errorDescriptor['tgtSpikeRate']
        startID = np.rint( tgtSpikeRegion['start'] / self.simulation['Ts'] ).astype(int)
        stopID  = np.rint( tgtSpikeRegion['stop' ] / self.simulation['Ts'] ).astype(int)

        desiredClass = desiredClass.cpu().data.numpy()
        actualSpikes = torch.sum(spikeOut[...,startID:stopID], 4, keepdim=True).cpu().detach().numpy() * self.simulation['Ts']
        desiredRate  = tgtSpikeRate[True] * desiredClass + tgtSpikeRate[False] * (1 - desiredClass)
        desiredSpikes = desiredRate * spikeOut.shape[-1]

        if earlySpikeBias is None:
            errorSpikeCount = (actualSpikes - desiredSpikes) / (stopID - startID) # * scale
        else:
            print('To be implemented.')

        targetRegion = np.zeros(spikeOut.shape)
        targetRegion[:,:,:,:,startID:stopID] = 1;
        spikeDesired = torch.FloatTensor(targetRegion * spikeOut.cpu().data.numpy()).to(spikeOut.device)
        
        # error = self.psp(spikeOut - spikeDesired)
        error = self.slayer.psp(spikeOut - spikeDesired)
        error += torch.FloatTensor(errorSpikeCount * targetRegion).to(spikeOut.device)
        
        return 1/2 * torch.sum(error**2) * self.simulation['Ts']

    def loss_mem(self, spikeOut, desiredClass, mem):
        return loss_mem.apply(spikeOut, desiredClass, mem)

    def probSpikes(self, spikeOut, desiredProb, probSlidingWindow=None):
        assert self.errorDescriptor['type'] == 'ProbSpikes', "Error type is not ProbSpikes"
        if probSlidingWindow is None:
            # return self._globalProbSpikes(spikeOut, desiredProb)
            return globalProbSpikes.apply(spikeOut, desiredProb)
        else:
            return runningProbSpikes.apply(spikeOut, desiredProb, probSlidingWindow)

    def adaptiveSpikes(self, spikeOut, desiredClass):
        assert self.errorDescriptor['type'] == 'AdaptiveSpikes', "Error type is not AdaptiveSpikes"
        
        tgtSpikeRate   = self.errorDescriptor['tgtSpikeRate']

        desiredClass = desiredClass.cpu().data.numpy()
        actualSpikes = torch.sum(spikeOut, 4, keepdim=True).cpu().detach().numpy() * self.simulation['Ts']
        desiredRate  = tgtSpikeRate[True] * desiredClass + tgtSpikeRate[False] * (1 - desiredClass)
        desiredSpikes = np.rint(desiredRate * spikeOut.shape[-1])

        spikeDes  = np.zeros(spikeOut.shape)

        N, C, H, W, T = spikeOut.shape

        for n in range(N):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        outAE = np.argwhere(spikeOut[n, c, h, w].cpu().data.numpy() > 0).flatten()
                        if actualSpikes[n, c, h, w] < desiredSpikes[n, c, h, w]:
                            deficitSpikes = int(desiredSpikes[n, c, h, w] - actualSpikes[n, c, h, w])
                            desAE = np.hstack((outAE, np.random.permutation(T)[:deficitSpikes]))
                            spikeDes[n, c, h, w, desAE] = 1 / self.simulation['Ts']
                        elif actualSpikes[n, c, h, w] > desiredSpikes[n, c, h, w]:
                            desAE = outAE[:int(desiredSpikes[n, c, h, w])] # only take the first spikes
                            spikeDes[n, c, h, w, desAE] = 1 / self.simulation['Ts']
                        else:
                            spikeDes[n, c, h, w, outAE] = 1 / self.simulation['Ts']

        error = self.slayer.psp(spikeOut - torch.FloatTensor(spikeDes).to(spikeOut.device)) 
        return 1/2 * torch.sum(error**2) * self.simulation['Ts']



    def variableRate(self, spikeOut, desiredRate, slidingWindow) :
        # assert self.errorDescriptor['type'] = 'TemporalRate', "Error type is not TemporalRate"
        pass

class gradLog(torch.autograd.Function):
    data = None
    # data = 

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, gradOutput):
        gradLog.data = gradOutput
        # gradLog.data.append(gradOutput)
        return gradOutput

class conv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, filter):
        psp = slayerCuda.conv(x.contiguous(), filter, 1)
        ctx.save_for_backward(filter)
        return psp

    @staticmethod
    def backward(ctx, gradOutput):
        '''
        '''
        (filter, ) = ctx.saved_tensors
        gradInput = slayerCuda.corr(gradOutput.contiguous(), filter, 1)
        
        return gradInput, None

class runningProbSpikes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spikeOut, desiredProb, probSlidingWindow):
        slidingWindow = torch.ones((probSlidingWindow), dtype=torch.float32, requires_grad=False)
        spikeCount = conv.apply(spikeOut.contiguous(), slidingWindow.to(spikeOut.device)).clamp(1)
        totalSpike = spikeCount.sum(dim=[1, 2, 3], keepdim=True)
        probability = spikeCount / totalSpike

        ctx.save_for_backward(probability, desiredProb, spikeCount)

        return torch.sum( -desiredProb * torch.log(probability))

    @staticmethod
    def backward(ctx, gradOutput):
        (probability, desiredProb, spikeCount) = ctx.saved_tensors
        
        grad = (probability - desiredProb) / spikeCount

        return grad * gradOutput, None, None

class globalProbSpikes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spikeOut, desiredProb):
        spikeCount  = torch.sum(spikeOut, 4, keepdim=True).clamp(1) # minimum spike count of 1 for numerical stability
        totalSpike  = spikeCount.sum(dim=[1, 2, 3], keepdim=True)
        probability = spikeCount / totalSpike

        T = torch.FloatTensor([spikeOut.shape[-1]]).to(spikeOut.device)

        # print(outShape)
        ctx.save_for_backward(probability, desiredProb, spikeCount, T)

        return torch.sum( -desiredProb * torch.log(probability))

    @staticmethod
    def backward(ctx, gradOutput):
        (probability, desiredProb, spikeCount, T) = ctx.saved_tensors
        N, C, H, W, _ = probability.shape
        
        grad = (probability - desiredProb) / spikeCount * torch.ones((N, C, H, W, int(T.item())), dtype=torch.float32, device=probability.device)

        return grad * gradOutput, None

class loss_mem(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spikeOut, desiredClass, mem):
        N, C, H, W, T = spikeOut.shape
        #spikeOutcopy = spikeOut.clone()
        desiredClasscpu = desiredClass.cpu().data.numpy()
        #desiredmem  = 1.05 * desiredClass*10 + 0.95 * (1 - desiredClass) * 10  #10 means theta
        desiredmem  = 0 * desiredClasscpu*10 + 0.9950 * (1 - desiredClasscpu) * 10  #10 means theta, don;t care the true neuron
        spikeOutcpu = spikeOut.cpu().detach().numpy() 
        #print('shape', desiredmem.shape) 
        desiredmem = desiredmem.repeat((spikeOutcpu.shape[-1]), axis=4)    
            
        desiredmem = desiredmem * spikeOutcpu

        #spikeOutmem = spikeOut * (mem.cpu().detach().numpy())
        spikeOutmem = spikeOutcpu * (mem.cpu().detach().numpy())

        mask =  desiredClasscpu*0  + 1 * (1 - desiredClasscpu)           #true class should not be depressed
        spikeOutmem = spikeOutmem * (mask.repeat((spikeOutcpu.shape[-1]), axis=4))
        desiredmem = desiredmem * (mask.repeat((spikeOutcpu.shape[-1]), axis=4))
        #print('mem ', desiredmem[1,1,0,0,600:620])
        spikeDesired = torch.FloatTensor((spikeOutmem-desiredmem))
        #print('desiredmem', desiredmem[1,0,0,0,:10])
        #print('spikeOutmem', spikeOutmem[1,0,0,0,:])
        spikeCount  = torch.sum(spikeOut, 4, keepdim=True).clamp(1) # minimum spike count of 1 for numerical stability
        #error = (spikeDesired.to(mem.device)/spikeCount)
        error = (spikeDesired.to(mem.device)/spikeOutcpu.shape[-1])
        ctx.save_for_backward(mem, error, spikeOut)        
        return 1/2 * torch.sum(error**2) 
    @staticmethod
    def backward(ctx, gradOutput):
        (mem, error, spikeOut) = ctx.saved_tensors
        grad = (error*spikeOut).to(mem.device) 
        return None, None, gradOutput*grad 
