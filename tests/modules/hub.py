import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from typing import Tuple, List, Dict

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

models = {
    "alexnet": {
        "model": models.alexnet(pretrained=True),
        "path": "both"
    },
    "vgg16": {
        "model": models.vgg16(pretrained=True),
        "path": "both"
    },
    "squeezenet": {
        "model": models.squeezenet1_0(pretrained=True),
        "path": "both"
    },
    "densenet": {
        "model": models.densenet161(pretrained=True),
        "path": "both"
    },
    "inception_v3": {
        "model": models.inception_v3(pretrained=True),
        "path": "both"
    },
    #"googlenet": models.googlenet(pretrained=True),
    "shufflenet": {
        "model": models.shufflenet_v2_x1_0(pretrained=True),
        "path": "both"
    },
    "mobilenet_v2": {
        "model": models.mobilenet_v2(pretrained=True),
        "path": "both"
    },
    "resnext50_32x4d": {
        "model": models.resnext50_32x4d(pretrained=True),
        "path": "both"
    },
    "wideresnet50_2": {
        "model": models.wide_resnet50_2(pretrained=True),
        "path": "both"
    },
    "mnasnet": {
        "model": models.mnasnet1_0(pretrained=True),
        "path": "both"
    },
    "resnet18": {
        "model": torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True),
        "path": "both"
    },
    "resnet50": {
        "model": torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True),
        "path": "both"
    },
    "ssd": {
        "model": torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math="fp32"),
        "path": "trace"
    },
    "efficientnet_b0": {
        "model": timm.create_model('efficientnet_b0', pretrained=True),
        "path": "script"
    },
    "vit": {
        "model": timm.create_model('vit_base_patch16_224', pretrained=True),
        "path": "script"
    }
}

# Download sample models
for n, m in models.items():
    print("Downloading {}".format(n))
    m["model"] = m["model"].eval().cuda()
    x = torch.ones((1, 3, 300, 300)).cuda()
    if m["path"] == "both" or m["path"] == "trace":
        trace_model = torch.jit.trace(m["model"], [x])
        torch.jit.save(trace_model, n + '_traced.jit.pt')
    if m["path"] == "both" or m["path"] == "script":
        script_model = torch.jit.script(m["model"])
        torch.jit.save(script_model, n + '_scripted.jit.pt')


# Sample Pool Model (for testing plugin serialization)
class Pool(nn.Module):

    def __init__(self):
        super(Pool, self).__init__()

    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (5, 5))


model = Pool().eval().cuda()
x = torch.ones([1, 3, 10, 10]).cuda()

trace_model = torch.jit.trace(model, x)
torch.jit.save(trace_model, "pooling_traced.jit.pt")


# Sample Nested Module (for module-level fallback testing)
class ModuleFallbackSub(nn.Module):

    def __init__(self):
        super(ModuleFallbackSub, self).__init__()
        self.conv = nn.Conv2d(1, 3, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class ModuleFallbackMain(nn.Module):

    def __init__(self):
        super(ModuleFallbackMain, self).__init__()
        self.layer1 = ModuleFallbackSub()
        self.conv = nn.Conv2d(3, 6, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(self.layer1(x)))


module_fallback_model = ModuleFallbackMain().eval().cuda()
module_fallback_script_model = torch.jit.script(module_fallback_model)
torch.jit.save(module_fallback_script_model, "module_fallback_scripted.jit.pt")


# Sample Looping Modules (for loop fallback testing)
class LoopFallbackEval(nn.Module):

    def __init__(self):
        super(LoopFallbackEval, self).__init__()

    def forward(self, x):
        add_list = torch.empty(0).to(x.device)
        for i in range(x.shape[1]):
            add_list = torch.cat((add_list, torch.tensor([x.shape[1]]).to(x.device)), 0)
        return x + add_list


class LoopFallbackNoEval(nn.Module):

    def __init__(self):
        super(LoopFallbackNoEval, self).__init__()

    def forward(self, x):
        for _ in range(x.shape[1]):
            x = x + torch.ones_like(x)
        return x


loop_fallback_eval_model = LoopFallbackEval().eval().cuda()
loop_fallback_eval_script_model = torch.jit.script(loop_fallback_eval_model)
torch.jit.save(loop_fallback_eval_script_model, "loop_fallback_eval_scripted.jit.pt")
loop_fallback_no_eval_model = LoopFallbackNoEval().eval().cuda()
loop_fallback_no_eval_script_model = torch.jit.script(loop_fallback_no_eval_model)
torch.jit.save(loop_fallback_no_eval_script_model, "loop_fallback_no_eval_scripted.jit.pt")


# Sample Conditional Model (for testing partitioning and fallback in conditionals)
class FallbackIf(torch.nn.Module):

    def __init__(self):
        super(FallbackIf, self).__init__()
        self.relu1 = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.log_sig = torch.nn.LogSigmoid()
        self.conv2 = torch.nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, x):
        x = self.relu1(x)
        x_first = x[0][0][0][0].item()
        if x_first > 0:
            x = self.conv1(x)
            x1 = self.log_sig(x)
            x2 = self.conv2(x)
            x = self.conv3(x1 + x2)
        else:
            x = self.log_sig(x)
        x = self.conv1(x)
        return x


conditional_model = FallbackIf().eval().cuda()
conditional_script_model = torch.jit.script(conditional_model)
torch.jit.save(conditional_script_model, "conditional_scripted.jit.pt")


# Collection input/output models
class Normal(nn.Module):
    def __init__(self):
        super(Normal, self).__init__()

    def forward(self, x, y):
        r = x + y
        return r

class TupleInput(nn.Module):
    def __init__(self):
        super(TupleInput, self).__init__()

    def forward(self, z: Tuple[torch.Tensor, torch.Tensor]):
        r = z[0] + z[1]
        return r

class ListInput(nn.Module):
    def __init__(self):
        super(ListInput, self).__init__()

    def forward(self, z: List[torch.Tensor]):
        r = z[0] + z[1]
        return r

class TupleInputOutput(nn.Module):
    def __init__(self):
        super(TupleInputOutput, self).__init__()

    def forward(self, z: Tuple[torch.Tensor, torch.Tensor]):
        r1 = z[0] + z[1]
        r2 = z[0] - z[1]
        r = (r1, r2)
        return r

class ListInputOutput(nn.Module):
    def __init__(self):
        super(ListInputOutput, self).__init__()

    def forward(self, z: List[torch.Tensor]):
        r1 = z[0] + z[1]
        r2 = z[0] - z[1]
        r = [r1, r2]
        return r

class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.list_model = ListInputOutput()
        self.tuple_model = TupleInputOutput()

    def forward(self, z: List[torch.Tensor]):
        r1 = z[0] + z[1]
        r2 = z[0] - z[1]
        r3 = (r1, r2)
        r4 = [r2, r1]
        tuple_out = self.tuple_model(r3)
        list_out = self.list_model(r4)
        r = (tuple_out[1], list_out[0])
        return r

normal_model = Normal()
normal_model_ts = torch.jit.script(normal_model)
normal_model_ts.to("cuda").eval()
torch.jit.save(normal_model_ts, "normal_model.jit.pt")

tuple_input = TupleInput()
tuple_input_ts = torch.jit.script(tuple_input)
tuple_input_ts.to("cuda").eval()
torch.jit.save(tuple_input_ts, "tuple_input.jit.pt")

list_input = ListInput()
list_input_ts = torch.jit.script(list_input)
list_input_ts.to("cuda").eval()
torch.jit.save(list_input_ts, "list_input.jit.pt")

tuple_input = TupleInputOutput()
tuple_input_ts = torch.jit.script(tuple_input)
tuple_input_ts.to("cuda").eval()
torch.jit.save(tuple_input_ts, "tuple_input_output.jit.pt")

list_input = ListInputOutput()
list_input_ts = torch.jit.script(list_input)
list_input_ts.to("cuda").eval()
torch.jit.save(list_input_ts, "list_input_output.jit.pt")

complex_model = ComplexModel()
complex_model_ts = torch.jit.script(complex_model)
complex_model_ts.to("cuda").eval()
torch.jit.save(complex_model_ts, "complex_model.jit.pt")