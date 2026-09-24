import torch


class EMA:
    def __init__(self, momentum: float = 0.999):
        assert 0.0 < momentum < 1.0
        self.m = momentum

    @torch.no_grad()
    def update(self, teacher, student):
        for (tn, tp), (sn, sp) in zip(teacher.named_parameters(), student.named_parameters()):
            if tp.data.shape == sp.data.shape:
                tp.data.lerp_(sp.data, 1.0 - self.m)

    @torch.no_grad()
    def copy(self, teacher, student):
        teacher.load_state_dict(student.state_dict(), strict=True)
        for p in teacher.parameters():
            p.requires_grad_(False)
