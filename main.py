pip install PyWavelets
pip install pycryptodome
pip install pywavelets pillow pycryptodome

import numpy as np
from PIL import Image
import pywt


def dwt_lsb_embed(image_path, binary_data, output_path):
    # Load grayscale image: Loads the image and converts it to grayscale ('L' mode).
    #Grayscale makes it simpler (1 channel instead of 3 RGB channels).

    img = Image.open(image_path).convert('L')
    #Converts the image to a NumPy array of floats, which is needed for performing
    #wavelet transforms.
    img_array = np.array(img, dtype=np.float32)

    # Apply DWT: DWT decomposes the image into 4 sub-bands:
    #LL: Approximation (low-frequency, contains most of the visual content)
    #LH, HL, HH: Detail bands (high-frequency)
    #We're embedding the data in the LL band, which is unusual but offers high capacity.
    #(Often, HH is preferred for less visible change.)
    coeffs2 = pywt.dwt2(img_array, 'haar')
    LL, (LH, HL, HH) = coeffs2

    # Flatten LL and embed binary data in LSB: Flattens the LL band (2D → 1D) so it’s easy to loop through pixel-wise.
    flat_LL = LL.flatten()
    if len(binary_data) > len(flat_LL):
        raise ValueError("❌ Not enough capacity in LL band for the message.")

    #For each pixel:
    #Converts to binary.
    #Replaces the last bit (LSB) with 1 bit of your secret message.
    #Converts it back to integer and updates the pixel.
    for i in range(len(binary_data)):
        pixel = int(flat_LL[i])
        pixel_bin = format(pixel, '08b')
        pixel_bin = pixel_bin[:-1] + binary_data[i]
        flat_LL[i] = int(pixel_bin, 2)

    # Reshape and reconstruct image using inverse DWT
    LL_embedded = flat_LL.reshape(LL.shape)
    coeffs2_embedded = (LL_embedded, (LH, HL, HH))
    stego_array = pywt.idwt2(coeffs2_embedded, 'haar')
    stego_array = np.clip(stego_array, 0, 255).astype(np.uint8)

    # Save the stego-image
    stego_img = Image.fromarray(stego_array)
    stego_img.save(output_path)
    print(f"✅ Embedded using DWT + LSB. Saved as {output_path}")


def dwt_lsb_extract(stego_image_path): #Loads the stego image and prepares it for wavelet processing.

    img = Image.open(stego_image_path).convert('L')
    img_array = np.array(img, dtype=np.float32)

    # Apply DWT: Performs DWT again and flattens the LL band to access the embedded bits.
    coeffs2 = pywt.dwt2(img_array, 'haar')
    LL, _ = coeffs2

    flat_LL = LL.flatten()
    extracted_bits = ""
    for pixel in flat_LL:
        pixel_bin = format(int(pixel), '08b')
        extracted_bits += pixel_bin[-1]
        # End marker: Detects the 16-bit end delimiter you appended during embedding.
        # If found → trimming it, then returns only the actual message bits.
        if extracted_bits.endswith('1111111111111110'):
            extracted_bits = extracted_bits[:-16]
            print(f"ℹ Extracted {len(extracted_bits)} bits before end marker.")
            return extracted_bits

    print("❌ End marker not found.")
    return None


from PIL import Image
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes

def aes_encrypt(plaintext, key):
    key = key.ljust(16, ' ').encode('utf-8')[:16]
    iv = get_random_bytes(AES.block_size)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded_text = pad(plaintext.encode('utf-8'), AES.block_size)
    ciphertext = cipher.encrypt(padded_text)
    return iv + ciphertext

def aes_decrypt(ciphertext, key):
    key = key.ljust(16, ' ').encode('utf-8')[:16]
    iv = ciphertext[:AES.block_size]
    ciphertext = ciphertext[AES.block_size:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_text = unpad(cipher.decrypt(ciphertext), AES.block_size).decode('utf-8')
    return decrypted_text

def embed_data_custom_bit(image_path, secret_message, key, output_path, bit_position=0):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img, dtype=np.int32)
    height, width, channels = img_array.shape

    encrypted = aes_encrypt(secret_message, key)
    binary_data = ''.join(format(byte, '08b') for byte in encrypted)
    end_marker = '1111111111111110'
    data_to_embed = binary_data + end_marker
    data_len = len(data_to_embed)

    print("Embedding:")
    print(f"Original message: {secret_message}")
    print(f"Encrypted message (in bytes): {encrypted}")
    print(f"Binary data to embed (first 50 bits): {data_to_embed[:50]}...")
    print(f"Bit Position: {bit_position}")
    print(f"Total bits to embed: {data_len}")

    if data_len > height * width * 3:
        raise ValueError("Data too large to embed in the image")

    data_index = 0
    for row in range(height):
        for col in range(width):
            for channel in range(channels):
                if data_index < data_len:
                    bit_mask = 1 << bit_position
                    img_array[row, col, channel] &= ~bit_mask  # Clear the target bit
                    img_array[row, col, channel] |= (int(data_to_embed[data_index]) << bit_position)
                    data_index += 1
                else:
                    break
            if data_index >= data_len:
                break
        if data_index >= data_len:
            break

    embedded_img_array = img_array.astype(np.uint8)
    embedded_img = Image.fromarray(embedded_img_array)
    embedded_img.save(output_path)
    print(f"Successfully embedded {data_len} bits in {output_path}")

def extract_data_custom_bit(stego_path, key, bit_position=0):
    stego_img = Image.open(stego_path).convert('RGB')
    stego_array = np.array(stego_img)
    height, width, channels = stego_array.shape

    extracted_bin = ""
    num_bits_to_extract = height * width * channels

    for row in range(height):
        for col in range(width):
            for channel in range(channels):
                if len(extracted_bin) < num_bits_to_extract:
                    bit_mask = 1 << bit_position
                    extracted_bit = (stego_array[row, col, channel] & bit_mask) >> bit_position
                    extracted_bin += str(extracted_bit)
                else:
                    break
            if len(extracted_bin) >= num_bits_to_extract:
                break
        if len(extracted_bin) >= num_bits_to_extract:
            break

    print("\nExtraction:")
    print(f"Extracted Binary Data Length: {len(extracted_bin)}")
    print(f"Extracted Binary Data (first 100 bits): {extracted_bin[:100]}...")

    end_marker_bin = '1111111111111110'
    end_marker_index = extracted_bin.find(end_marker_bin)

    if end_marker_index != -1:
        print(f"End marker found at bit index: {end_marker_index}")
        extracted_data_bin = extracted_bin[:end_marker_index]
        extracted_bytes = [
            int(extracted_data_bin[i:i+8], 2)
            for i in range(0, len(extracted_data_bin), 8)
            if len(extracted_data_bin[i:i+8]) == 8
        ]
        try:
            decrypted_data = aes_decrypt(bytes(extracted_bytes), key)
            print("Extracted Message:", decrypted_data)
            return decrypted_data
        except Exception as e:
            print(f"Decryption error: {e}")
            return None
    else:
        print("End marker not found.")
        return None

# Example main function
if _name_ == "_main_":
    input_img = "input_image.png"
    output_img = "stego_custombit.png"
    message = "Confidential: Patient diagnosed with hypertension"
    encryption_key = "medkey123"
    bit_position = 5  # Change this to embed/extract from a different bit (0 to 7)

    try:
        embed_data_custom_bit(input_img, message, encryption_key, output_img, bit_position)
        recovered_message = extract_data_custom_bit(output_img, encryption_key, bit_position)
        if recovered_message:
            print("Recovered Message:", recovered_message)
    except FileNotFoundError:
        print(f"Error: Input image '{input_img}' not found.")
    except Exception as e:
        print(f"Error: {e}")
      
