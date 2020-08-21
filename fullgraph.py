import torch

class FullGraph(torch.nn.Module):
    def __init__(self, resnet_graph, ale_graph, classVectors):

        super(FullGraph, self).__init__()
        self.resnet_graph = resnet_graph
        self.ale_graph = ale_graph
        self.classVectors =classVectors

    def forward(self, image):

        features = self.resnet_graph(image)
        features = features.squeeze(2).squeeze(2)
        out =  self.ale_graph(features, self.classVectors)

        return out