import torch as t
import time
import torch.utils.model_zoo as model_zoo

class BasicModule(t.nn.Module):
    """
    封装nn.Module，主要提供save和load两个方法
    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def init_pretrained_weights(self, model_url):
        """Initializes model with pretrained weights.

            Layers that don't match with pretrained layers in name or size are kept unchanged.
            """
        if model_url[:3] == "http":
            pretrain_dict = model_zoo.load_url(model_url)
        else:
            pretrain_dict = t.load(model_url)
        model_dict = self.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        self.load_state_dict(model_dict)
        print('Initialized model with pretrained weights from {}'.format(model_url))

    def load(self, path, use_gpu=False):
        """
        可加载指定路径的模型
        """
        if not use_gpu:
            self.load_state_dict(t.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用"模型名字+时间"作为文件名
        """
        if name is None:
            prefix = self.module_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), 'checkpoints/' + name)
        return name

    def forward(self, *input):
        pass

