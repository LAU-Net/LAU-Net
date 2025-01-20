import numpy as np
from pesq import pesq
from metric_helper import wss, llr, SSNR, trim_mos
# https://github.com/wooseok-shin/MetricGAN-plus-pytorch/blob/main/metric_functions/compute_metric.py

def PESQ_normalize(x):
    # Obtained from: https://github.com/nii-yamagishilab/NELE-GAN/blob/master/intel.py (def mapping_PESQ_harvard)
    a = -1.5
    b = 2.5
    y = 1/(1+np.exp(a *(x - b)))
    return y

# def PESQ_normalize(x):
#     y = (x + 0.5) / 5
#     return y

def CMOS_normalize(x):
    y = (x - 1.0) / 4
    return y

def compute_pesq(target_wav, pred_wav, fs, norm=False):
    # Compute the PESQ
    Pesq = pesq(fs, target_wav, pred_wav, 'nb')

    if norm:
        return PESQ_normalize(Pesq)
    else:
        return Pesq

def compute_csig(target_wav, pred_wav, fs, norm=False):
    alpha   = 0.95

    # Compute WSS measure
    wss_dist_vec = wss(target_wav, pred_wav, fs)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist     = np.mean(wss_dist_vec[:int(round(len(wss_dist_vec) * alpha))])

    # Compute LLR measure
    LLR_dist = llr(target_wav, pred_wav, fs)
    LLR_dist = sorted(LLR_dist, reverse=False)
    LLRs     = LLR_dist
    LLR_len  = round(len(LLR_dist) * alpha)
    llr_mean = np.mean(LLRs[:LLR_len])

    # Compute the PESQ
    pesq_raw = pesq(fs, target_wav, pred_wav, 'nb')

    # Csig
    Csig = 3.093 - 1.029 * llr_mean + 0.603 * pesq_raw - 0.009 * wss_dist
    # print("Csig: ", Csig)
    # print("LLR: ", llr_mean)
    # print("PESQ: ", pesq_raw)
    # print("WSS: ", wss_dist)

    Csig = float(trim_mos(Csig))
    
    if norm:
        return CMOS_normalize(Csig)
    else:
        return Csig

def compute_cbak(target_wav, pred_wav, fs, norm=False):
    alpha   = 0.95

    # Compute WSS measure
    wss_dist_vec = wss(target_wav, pred_wav, fs)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist     = np.mean(wss_dist_vec[:int(round(len(wss_dist_vec) * alpha))])
    
    # Compute the SSNR
    snr_mean, segsnr_mean = SSNR(target_wav, pred_wav, fs)
    segSNR = np.mean(segsnr_mean)

    # Compute the PESQ
    pesq_raw = pesq(fs, target_wav, pred_wav, 'nb')

    # Cbak
    Cbak = 1.634 + 0.478 * pesq_raw - 0.007 * wss_dist + 0.063 * segSNR
    Cbak = trim_mos(Cbak)
    
    if norm:
        return CMOS_normalize(Cbak)
    else:    
        return Cbak

def compute_covl(target_wav, pred_wav, fs, norm=False):
    alpha   = 0.95

    # Compute WSS measure
    wss_dist_vec = wss(target_wav, pred_wav, fs)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist     = np.mean(wss_dist_vec[:int(round(len(wss_dist_vec) * alpha))])

    # Compute LLR measure
    LLR_dist = llr(target_wav, pred_wav, fs)
    LLR_dist = sorted(LLR_dist, reverse=False)
    LLRs     = LLR_dist
    LLR_len  = round(len(LLR_dist) * alpha)
    llr_mean = np.mean(LLRs[:LLR_len])

    # Compute the PESQ
    pesq_raw = pesq(fs, target_wav, pred_wav, 'nb')

    # Covl
    Covl = 1.594 + 0.805 * pesq_raw - 0.512 * llr_mean - 0.007 * wss_dist
    Covl = trim_mos(Covl)

    if norm:
        return CMOS_normalize(Covl)
    else:
        return Covl
    

def si_sdr(reference, estimation):
    # https://github.com/fgnt/pb_bss/blob/master/pb_bss/evaluation/module_si_sdr.py
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)

    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]

    Returns:
        SI-SDR

    [1] SDR– Half- Baked or Well Done?
    http://www.merl.com/publications/docs/TR2019-013.pdf

    >>> np.random.seed(0)
    >>> reference = np.random.randn(100)
    >>> si_sdr(reference, reference)
    inf
    >>> si_sdr(reference, reference * 2)
    inf
    >>> si_sdr(reference, np.flip(reference))
    -25.127672346460717
    >>> si_sdr(reference, reference + np.flip(reference))
    0.481070445785553
    >>> si_sdr(reference, reference + 0.5)
    6.3704606032577304
    >>> si_sdr(reference, reference * 2 + 1)
    6.3704606032577304
    >>> si_sdr([1., 0], [0., 0])  # never predict only zeros
    nan
    >>> si_sdr([reference, reference], [reference * 2 + 1, reference * 1 + 0.5])
    array([6.3704606, 6.3704606])

    """
    estimation, reference = np.broadcast_arrays(estimation, reference)

    assert reference.dtype == np.float64, reference.dtype
    assert estimation.dtype == np.float64, estimation.dtype

    reference_energy = np.sum(reference ** 2, axis=-1, keepdims=True)

    # This is $\alpha$ after Equation (3) in [1].
    optimal_scaling = np.sum(reference * estimation, axis=-1, keepdims=True) \
        / reference_energy

    # This is $e_{\text{target}}$ in Equation (4) in [1].
    projection = optimal_scaling * reference

    # This is $e_{\text{res}}$ in Equation (4) in [1].
    noise = estimation - projection

    ratio = np.sum(projection ** 2, axis=-1) / np.sum(noise ** 2, axis=-1)
    return 10 * np.log10(ratio)