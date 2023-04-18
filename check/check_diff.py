from reprod_log import ReprodDiffHelper


def check_diff(file1, file2, out_file, tresh=1e-5):
    diff_helper = ReprodDiffHelper()

    info1 = diff_helper.load_info(file1)
    info2 = diff_helper.load_info(file2)

    diff_helper.compare_info(info1, info2)

    diff_helper.report(
        diff_method="mean", diff_threshold=tresh, path=out_file)


def check_lr():
    check_diff("check/backward/pd/lr_pd.npy", "check/backward/th/lr_th.npy", "./diff_lr.log")

def check_backward():
    check_diff("/home/xiejunlin/workspace/IBOT-Paddle/reprod_out/pd_backward.npy", "/home/xiejunlin/workspace/ibot/reprod_out/th_backward.npy", "/home/xiejunlin/workspace/IBOT-Paddle/check/diff_backward.log")


def check_forward_dino_loss():
    check_diff("./pd/forward_dino_loss_pd.npy", "./th/forward_dino_loss_th.npy", "./dino_loss_diff.log")


if __name__ == "__main__":
    # check_lr()
    check_backward()
    # check_forward_dino_loss()