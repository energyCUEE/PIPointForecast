from utils.formulations import PIloss_split, Regressionloss


def get_loss_functions(threshold, desired_picp):
    pi_loss = PIloss_split(
        threshold=threshold,
        dev_picp=None,
        desired_picp_night=0.5,
        mul_factor=10,  # 20
        delta=1 - desired_picp,
        soften=80,
        piwidth="sumk",
        split_piwidth=False,
        k=0.3,
        lmbda=0.8,
    )
    reg_loss = Regressionloss()  # MAE
    return pi_loss, reg_loss
