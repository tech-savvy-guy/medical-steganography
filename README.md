# _Secure Medical Image Steganography using DWT, AES, and LSB_

This project demonstrates a secure and robust steganography technique for embedding confidential patient information inside medical images using a combination of:

- *AES Encryption* for confidentiality,
- *LSB (Least Significant Bit) embedding* for imperceptibility,
- *DWT (Discrete Wavelet Transform)* for frequency-domain robustness.

---

## Features

- AES encryption of the secret patient message.
- DWT transformation of input images to identify embedding zones.
- LSB embedding in the frequency domain (HH sub-band).
- End-marker detection for accurate message extraction.
- Performance evaluation using:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
  - BER (Bit Error Rate)

---

## Dependencies

Install the required libraries using pip:

```bash
pip install numpy pillow pycryptodome scikit-image pywavelets
```
