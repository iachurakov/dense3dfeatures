import torch


class ConditionalNegatives:
    def __init__(self,
                 lower_bound: float,
                 upper_bound: float,
                 final_lower_bound: float,
                 final_upper_bound: float,
                 saturation_epoch_num: int
                 ):
        """
        :param lower_bound: lower percentile
        :param upper_bound: upper percentile
        :param final_lower_bound: final value of lower bound
        :param final_upper_bound: final value of upper bound
        :param saturation_epoch_num: the number
         of epoch when upper_bound will be equal to final_upper_bound
        """

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.saturation_epoch_num = saturation_epoch_num
        self.current_epoch = 0
        self.cur_lower_bound, self.cur_upper_bound = 0, 0
        self.increase_factor = (final_lower_bound - lower_bound) / (saturation_epoch_num - 1)
        self.decrease_factor = (final_upper_bound - upper_bound) / (saturation_epoch_num - 1)

    def step(self):
        if self.current_epoch > self.saturation_epoch_num:
            return

        self.cur_lower_bound = self.lower_bound + self.current_epoch * self.increase_factor
        self.cur_upper_bound = self.upper_bound + self.current_epoch * self.decrease_factor

        self.current_epoch += 1

    def sample(self, logits, dim=1):
        sorted_logits = logits.sort(dim=dim)[0]
        lo = int(logits.shape[-1] * self.cur_lower_bound)
        hi = int(logits.shape[-1] * self.cur_upper_bound)
        if dim == 1:
            return sorted_logits[:, lo:hi]

        return torch.index_select(sorted_logits, dim, torch.arange(lo, hi))
