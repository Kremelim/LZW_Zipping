import os
import math
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from collections import Counter

# LZWCoding class for text and base compression
class LZWCoding:
    def __init__(self, filename, data_type, filepath=None):
        self.filename = filename
        self.data_type = data_type
        self.filepath = filepath if filepath else os.path.join(os.path.dirname(__file__), 
            filename + ('.txt' if data_type == 'text' else ''))
        self.codelength = None  # Will be updated during encoding

    @staticmethod
    def compute_entropy(data_str):
        """Calculate entropy of a string in bits per symbol."""
        if not data_str:
            return 0.0
        counter = Counter(data_str)
        total = len(data_str)
        return -sum((count / total) * math.log2(count / total) for count in counter.values() if count > 0)

    def compress_text_file(self):
        input_path = self.filepath
        directory = os.path.dirname(input_path)
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_file = base + '.bin'
        output_path = os.path.join(directory, output_file)

        try:
            with open(input_path, 'r') as in_file:
                text = in_file.read().rstrip()
        except FileNotFoundError:
            error_msg = f"Error: Input file '{input_path}' not found."
            print(error_msg)
            return None, error_msg
        except IOError as e:
            error_msg = f"Error reading file '{input_path}': {str(e)}"
            print(error_msg)
            return None, error_msg

        entropy = self.compute_entropy(text)
        encoded_integers = self.encode(text)
        avg_code_length = (len(encoded_integers) * self.codelength) / len(text) if text else 0
        encoded_bitstring = self.int_list_to_binary_string(encoded_integers)
        # Prepend 8-bit code length info as required (Level 1 actions-details)
        encoded_bitstring = self.add_code_length_info(encoded_bitstring)
        padded_bitstring = self.pad_encoded_data(encoded_bitstring)
        byte_array = self.get_byte_array(padded_bitstring)

        try:
            with open(output_path, 'wb') as out_file:
                out_file.write(bytes(byte_array))
        except IOError as e:
            error_msg = f"Error writing to file '{output_path}': {str(e)}"
            print(error_msg)
            return None, error_msg

        uncompressed_size = len(text)
        compressed_size = len(byte_array)
        ratio = uncompressed_size / compressed_size if compressed_size else 0
        stats = (f"Text file '{os.path.basename(input_path)}' compressed into '{output_file}'.\n"
                 f"Uncompressed Size: {uncompressed_size} bytes\n"
                 f"Compressed Size: {compressed_size} bytes\n"
                 f"Compression Ratio: {ratio:.2f}\n"
                 f"Entropy: {entropy:.2f} bits/symbol\n"
                 f"Avg Code Length: {avg_code_length:.2f} bits/symbol\n")
        print(stats)
        return output_path, stats

    def decompress_text_file(self):
        input_path = self.filepath
        directory = os.path.dirname(input_path)
        base = os.path.splitext(os.path.basename(input_path))[0]
        if base.endswith("_decompressed"):
            base = base[:-13]
        output_file = base + '_decompressed.txt'
        output_path = os.path.join(directory, output_file)

        try:
            with open(input_path, 'rb') as in_file:
                bytes_data = in_file.read()
        except FileNotFoundError:
            error_msg = f"Error: Compressed file '{input_path}' not found."
            print(error_msg)
            return None, error_msg
        except IOError as e:
            error_msg = f"Error reading file '{input_path}': {str(e)}"
            print(error_msg)
            return None, error_msg

        bit_string = "".join(bin(byte)[2:].rjust(8, '0') for byte in bytes_data)
        bit_string = self.remove_padding(bit_string)
        # Extract code length info (first 8 bits) as stored during compression
        bit_string = self.extract_code_length_info(bit_string)
        encoded_integers = self.binary_string_to_int_list(bit_string, self.codelength)
        decompressed_text = self.decode(encoded_integers)

        try:
            with open(output_path, 'w') as out_file:
                out_file.write(decompressed_text)
        except IOError as e:
            error_msg = f"Error writing to file '{output_path}': {str(e)}"
            print(error_msg)
            return None, error_msg

        stats = f"Compressed file '{os.path.basename(input_path)}' decompressed into '{output_file}'.\n"
        print(stats)
        return output_path, stats

    def compare_text_files(self, original_path, decompressed_path):
        """Compare original and decompressed text files"""
        try:
            with open(original_path, 'r') as orig_file:
                original_text = orig_file.read().rstrip()
            with open(decompressed_path, 'r') as dec_file:
                decompressed_text = dec_file.read().rstrip()
            
            is_identical = original_text == decompressed_text
            differences = sum(a != b for a, b in zip(original_text, decompressed_text))
            total_chars = max(len(original_text), len(decompressed_text))
            diff_percentage = (differences / total_chars * 100) if total_chars > 0 else 0
            
            stats = (f"Text File Comparison:\n"
                    f"Original: {os.path.basename(original_path)}\n"
                    f"Decompressed: {os.path.basename(decompressed_path)}\n"
                    f"Files identical: {is_identical}\n"
                    f"Differences: {differences} characters\n"
                    f"Difference Percentage: {diff_percentage:.2f}%\n"
                    f"Original length: {len(original_text)} chars\n"
                    f"Decompressed length: {len(decompressed_text)} chars\n")
            print(stats)
            return stats
        except Exception as e:
            error_msg = f"Error comparing text files: {str(e)}"
            print(error_msg)
            return error_msg

    def encode(self, uncompressed_data):
        dict_size = 256
        dictionary = {chr(i): i for i in range(dict_size)}
        w = ''
        result = []
        for k in uncompressed_data:
            wk = w + k
            if wk in dictionary:
                w = wk
            else:
                result.append(dictionary[w])
                dictionary[wk] = dict_size
                dict_size += 1
                w = k
        if w:
            result.append(dictionary[w])
        self.codelength = math.ceil(math.log2(len(dictionary)))
        return result

    def int_list_to_binary_string(self, int_list):
        return ''.join(format(num, f'0{self.codelength}b') for num in int_list)

    def add_code_length_info(self, bitstring):
        # Prepend the 8-bit code length information (as per project instructions)
        return f'{self.codelength:08b}' + bitstring

    def pad_encoded_data(self, encoded_data):
        extra_bits = (8 - len(encoded_data) % 8) % 8
        return f'{extra_bits:08b}' + encoded_data + '0' * extra_bits

    def get_byte_array(self, padded_encoded_data):
        if len(padded_encoded_data) % 8 != 0:
            raise ValueError('Data is not padded properly!')
        return bytearray(int(padded_encoded_data[i:i+8], 2) for i in range(0, len(padded_encoded_data), 8))

    def remove_padding(self, padded_encoded_data):
        extra_padding = int(padded_encoded_data[:8], 2)
        return padded_encoded_data[8:-extra_padding] if extra_padding else padded_encoded_data[8:]

    def extract_code_length_info(self, bitstring):
        self.codelength = int(bitstring[:8], 2)
        return bitstring[8:]

    def binary_string_to_int_list(self, bitstring, code_length):
        return [int(bitstring[i:i+code_length], 2) for i in range(0, len(bitstring), code_length)]

    def decode(self, compressed):
        dict_size = 256
        dictionary = {i: chr(i) for i in range(dict_size)}
        w = chr(compressed.pop(0))
        result = [w]
        for k in compressed:
            entry = dictionary[k] if k in dictionary else w + w[0] if k == dict_size else None
            if entry is None:
                raise ValueError(f"Bad compressed k: {k}")
            result.append(entry)
            dictionary[dict_size] = w + entry[0]
            dict_size += 1
            w = entry
        return ''.join(result)

# LZWImageCoding class for image compression
class LZWImageCoding(LZWCoding):
    def compress_image_file(self, image_format='bmp'):
        input_path = self.filepath
        directory = os.path.dirname(input_path)
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_file = base + '_img.bin'
        output_path = os.path.join(directory, output_file)

        try:
            img = Image.open(input_path)
        except FileNotFoundError:
            error_msg = f"Error: Image file '{input_path}' not found."
            print(error_msg)
            return None, error_msg
        except Exception as e:
            error_msg = f"Error opening image '{input_path}': {str(e)}"
            print(error_msg)
            return None, error_msg

        mode = img.mode
        width, height = img.size
        channels = img.split() if mode == 'RGB' else [img] if mode == 'L' else None
        if not channels:
            raise ValueError("Unsupported image mode")

        compressed_data = []
        stats = []
        for i, channel in enumerate(channels):
            channel_bytes = channel.tobytes()
            channel_str = channel_bytes.decode('latin1')
            entropy = self.compute_entropy(channel_str)
            encoded_integers = self.encode(channel_str)
            code_length = self.codelength
            avg_code_length = (len(encoded_integers) * code_length) / len(channel_str) if channel_str else 0
            stats.append((entropy, avg_code_length))
            encoded_bitstring = self.int_list_to_binary_string(encoded_integers)
            # For raw image compression we do not prepend code length info
            padded_bitstring = self.pad_encoded_data(encoded_bitstring)
            byte_array = self.get_byte_array(padded_bitstring)
            compressed_data.append((code_length, byte_array))

        try:
            with open(output_path, 'wb') as out_file:
                header = f"{mode},{width},{height}\n".encode('ascii')
                out_file.write(header)
                for code_length, byte_array in compressed_data:
                    out_file.write(code_length.to_bytes(1, 'big'))
                    out_file.write(len(byte_array).to_bytes(4, 'big'))
                    out_file.write(bytes(byte_array))
        except IOError as e:
            error_msg = f"Error writing to file '{output_path}': {str(e)}"
            print(error_msg)
            return None, error_msg

        total_compressed_size = sum(len(byte_array) for _, byte_array in compressed_data)
        original_size = width * height * (3 if mode == 'RGB' else 1)
        ratio = original_size / total_compressed_size if total_compressed_size else 0
        stats_msg = (f"Image file '{os.path.basename(input_path)}' compressed into '{output_file}'.\n"
                     f"Original Size: {original_size} bytes\n"
                     f"Compressed Size: {total_compressed_size} bytes\n"
                     f"Compression Ratio: {ratio:.2f}\n")
        for i, (entropy, avg_code_length) in enumerate(stats):
            channel_name = ['Red', 'Green', 'Blue'][i] if mode == 'RGB' else 'Gray'
            stats_msg += (f"{channel_name} Channel Entropy: {entropy:.2f} bits/symbol\n"
                          f"{channel_name} Channel Avg Code Length: {avg_code_length:.2f} bits/symbol\n")
        print(stats_msg)
        return output_path, stats_msg

    def decompress_image_file(self, image_format='bmp'):
        input_path = self.filepath
        directory = os.path.dirname(input_path)
        base = os.path.splitext(os.path.basename(input_path))[0]
        if base.endswith('_img'):
            base = base[:-4]
        output_file = base + '_img_decompressed.' + image_format
        output_path = os.path.join(directory, output_file)

        try:
            with open(input_path, 'rb') as in_file:
                header_line = in_file.readline().decode('ascii').strip()
                mode, width, height = header_line.split(',')
                width, height = int(width), int(height)
                num_channels = 3 if mode == 'RGB' else 1 if mode == 'L' else None
                if num_channels is None:
                    raise ValueError("Unsupported mode")

                channel_data = []
                for _ in range(num_channels):
                    code_length = int.from_bytes(in_file.read(1), 'big')
                    data_length = int.from_bytes(in_file.read(4), 'big')
                    byte_array = in_file.read(data_length)
                    bit_string = "".join(bin(byte)[2:].rjust(8, '0') for byte in byte_array)
                    bit_string = self.remove_padding(bit_string)
                    # For raw image decompression code length info was not prepended
                    encoded_integers = self.binary_string_to_int_list(bit_string, code_length)
                    decompressed_str = self.decode(encoded_integers)
                    channel_data.append(decompressed_str.encode('latin1'))

                img = (Image.frombytes('L', (width, height), channel_data[0]) if mode == 'L' else
                       Image.merge('RGB', [Image.frombytes('L', (width, height), data) for data in channel_data]))
        except FileNotFoundError:
            error_msg = f"Error: Compressed file '{input_path}' not found."
            print(error_msg)
            return None, error_msg
        except IOError as e:
            error_msg = f"Error reading file '{input_path}': {str(e)}"
            print(error_msg)
            return None, error_msg
        except Exception as e:
            error_msg = f"Error processing file '{input_path}': {str(e)}"
            print(error_msg)
            return None, error_msg

        try:
            img.save(output_path, image_format)
        except IOError as e:
            error_msg = f"Error saving image to '{output_path}': {str(e)}"
            print(error_msg)
            return None, error_msg

        stats = f"Compressed image file '{os.path.basename(input_path)}' decompressed into '{output_file}'.\n"
        print(stats)
        return output_path, stats

    def compress_gray_image_diff(self, image_format='bmp'):
        input_path = self.filepath
        directory = os.path.dirname(input_path)
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_file = base + '_graydiff.bin'
        output_path = os.path.join(directory, output_file)

        try:
            img = Image.open(input_path).convert('L')
        except FileNotFoundError:
            error_msg = f"Error: Image file '{input_path}' not found."
            print(error_msg)
            return None, error_msg
        except Exception as e:
            error_msg = f"Error opening image '{input_path}': {str(e)}"
            print(error_msg)
            return None, error_msg

        width, height = img.size
        img_array = np.array(img, dtype=np.int16)
        diff_array = np.zeros_like(img_array)
        # For each row: keep first pixel then take differences for subsequent pixels.
        for i in range(height):
            diff_array[i, 0] = img_array[i, 0]
            for j in range(1, width):
                diff_array[i, j] = img_array[i, j] - img_array[i, j-1]
        # For the first column (except the very first pixel), use column differences.
        for i in range(1, height):
            diff_array[i, 0] = img_array[i, 0] - img_array[i-1, 0]

        diff_list = diff_array.flatten().tolist()
        diff_str = ','.join(map(str, diff_list))
        entropy = self.compute_entropy(diff_str)
        header = f"L,{width},{height}"
        combined_text = header + "||" + diff_str
        encoded_integers = self.encode(combined_text)
        avg_code_length = (len(encoded_integers) * self.codelength) / len(combined_text) if combined_text else 0
        encoded_bitstring = self.int_list_to_binary_string(encoded_integers)
        # FIX: Prepend code length info so that decompression can extract it
        encoded_bitstring = self.add_code_length_info(encoded_bitstring)
        padded_bitstring = self.pad_encoded_data(encoded_bitstring)
        byte_array = self.get_byte_array(padded_bitstring)

        try:
            with open(output_path, 'wb') as out_file:
                out_file.write(bytes(byte_array))
        except IOError as e:
            error_msg = f"Error writing to file '{output_path}': {str(e)}"
            print(error_msg)
            return None, error_msg

        uncompressed_size = width * height
        compressed_size = len(byte_array)
        ratio = uncompressed_size / compressed_size if compressed_size else 0
        stats = (f"Grayscale image '{os.path.basename(input_path)}' compressed (difference method) into '{output_file}'.\n"
                 f"Original Size: {uncompressed_size} bytes\n"
                 f"Compressed Size: {compressed_size} bytes\n"
                 f"Compression Ratio: {ratio:.2f}\n"
                 f"Entropy of Difference Data: {entropy:.2f} bits/symbol\n"
                 f"Avg Code Length: {avg_code_length:.2f} bits/symbol\n")
        print(stats)
        return output_path, stats

    def decompress_gray_image_diff(self, image_format='bmp'):
        input_path = self.filepath
        directory = os.path.dirname(input_path)
        base = os.path.splitext(os.path.basename(input_path))[0].replace("_graydiff", "")
        output_file = base + '_graydiff_decompressed.' + image_format
        output_path = os.path.join(directory, output_file)

        try:
            with open(input_path, 'rb') as in_file:
                file_bytes = in_file.read()
            bit_string = "".join(bin(byte)[2:].rjust(8, '0') for byte in file_bytes)
            bit_string = self.remove_padding(bit_string)
            # Extract the code length info (first 8 bits)
            bit_string = self.extract_code_length_info(bit_string)
            encoded_integers = self.binary_string_to_int_list(bit_string, self.codelength)
            combined_text = self.decode(encoded_integers)
            header, diff_str = combined_text.split("||", 1)
            mode, width, height = header.split(',')
            width, height = int(width), int(height)
            diff_list = list(map(int, diff_str.split(',')))
            diff_array = np.array(diff_list, dtype=np.int16).reshape((height, width))
            img_array = np.zeros_like(diff_array)
            # Reconstruct the first column
            img_array[0, 0] = diff_array[0, 0]
            for i in range(1, height):
                img_array[i, 0] = img_array[i-1, 0] + diff_array[i, 0]
            # Reconstruct rows
            for i in range(height):
                for j in range(1, width):
                    img_array[i, j] = img_array[i, j-1] + diff_array[i, j]
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
        except Exception as e:
            error_msg = f"Error processing file '{input_path}': {str(e)}"
            print(error_msg)
            return None, error_msg

        try:
            img.save(output_path, image_format)
        except IOError as e:
            error_msg = f"Error saving image to '{output_path}': {str(e)}"
            print(error_msg)
            return None, error_msg

        stats = f"Grayscale difference file '{os.path.basename(input_path)}' decompressed into '{output_file}'.\n"
        print(stats)
        return output_path, stats

    def compress_color_image_diff(self, image_format='bmp'):
        input_path = self.filepath
        directory = os.path.dirname(input_path)
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_file = base + '_colordiff.bin'
        output_path = os.path.join(directory, output_file)

        try:
            img = Image.open(input_path).convert('RGB')
        except FileNotFoundError:
            error_msg = f"Error: Image file '{input_path}' not found."
            print(error_msg)
            return None, error_msg
        except Exception as e:
            error_msg = f"Error opening image '{input_path}': {str(e)}"
            print(error_msg)
            return None, error_msg

        width, height = img.size
        channels = img.split()
        compressed_data = []
        stats = []

        for i, channel in enumerate(channels):
            img_array = np.array(channel, dtype=np.int16)
            diff_array = np.zeros_like(img_array)
            # For each row: keep first pixel then take differences
            for row in range(height):
                diff_array[row, 0] = img_array[row, 0]
                for col in range(1, width):
                    diff_array[row, col] = img_array[row, col] - img_array[row, col-1]
            # For first column (from second row onward): column differences
            for row in range(1, height):
                diff_array[row, 0] = img_array[row, 0] - img_array[row-1, 0]

            diff_list = diff_array.flatten().tolist()
            diff_str = ','.join(map(str, diff_list))
            entropy = self.compute_entropy(diff_str)
            header = f"{i},{width},{height}"
            combined_text = header + "||" + diff_str
            encoded_integers = self.encode(combined_text)
            code_length = self.codelength
            avg_code_length = (len(encoded_integers) * code_length) / len(combined_text) if combined_text else 0
            stats.append((entropy, avg_code_length))
            encoded_bitstring = self.int_list_to_binary_string(encoded_integers)
            # For color difference, we write code length separately, so no need to add code length info here.
            padded_bitstring = self.pad_encoded_data(encoded_bitstring)
            byte_array = self.get_byte_array(padded_bitstring)
            compressed_data.append((code_length, byte_array))

        try:
            with open(output_path, 'wb') as out_file:
                header = f"RGB,{width},{height}\n".encode('ascii')
                out_file.write(header)
                for code_length, byte_array in compressed_data:
                    out_file.write(code_length.to_bytes(1, 'big'))
                    out_file.write(len(byte_array).to_bytes(4, 'big'))
                    out_file.write(bytes(byte_array))
        except IOError as e:
            error_msg = f"Error writing to file '{output_path}': {str(e)}"
            print(error_msg)
            return None, error_msg

        total_compressed_size = sum(len(byte_array) for _, byte_array in compressed_data)
        original_size = width * height * 3
        ratio = original_size / total_compressed_size if total_compressed_size else 0
        stats_msg = (f"Color image '{os.path.basename(input_path)}' compressed (difference method) into '{output_file}'.\n"
                     f"Original Size: {original_size} bytes\n"
                     f"Compressed Size: {total_compressed_size} bytes\n"
                     f"Compression Ratio: {ratio:.2f}\n")
        for i, (entropy, avg_code_length) in enumerate(stats):
            channel_names = ['Red', 'Green', 'Blue']
            stats_msg += (f"{channel_names[i]} Channel Entropy: {entropy:.2f} bits/symbol\n"
                          f"{channel_names[i]} Channel Avg Code Length: {avg_code_length:.2f} bits/symbol\n")
        print(stats_msg)
        return output_path, stats_msg

    def decompress_color_image_diff(self, image_format='bmp'):
        input_path = self.filepath
        directory = os.path.dirname(input_path)
        base = os.path.splitext(os.path.basename(input_path))[0].replace("_colordiff", "")
        output_file = base + '_colordiff_decompressed.' + image_format
        output_path = os.path.join(directory, output_file)

        try:
            with open(input_path, 'rb') as in_file:
                header_line = in_file.readline().decode('ascii').strip()
                mode, width, height = header_line.split(',')
                width, height = int(width), int(height)
                if mode != 'RGB':
                    raise ValueError("Expected RGB mode for color difference decompression")

                channel_data = []
                for _ in range(3):
                    code_length = int.from_bytes(in_file.read(1), 'big')
                    data_length = int.from_bytes(in_file.read(4), 'big')
                    byte_array = in_file.read(data_length)
                    bit_string = "".join(bin(byte)[2:].rjust(8, '0') for byte in byte_array)
                    bit_string = self.remove_padding(bit_string)
                    encoded_integers = self.binary_string_to_int_list(bit_string, code_length)
                    combined_text = self.decode(encoded_integers)
                    # The header for each channel is not needed for decompression here
                    _, diff_str = combined_text.split("||", 1)
                    diff_list = list(map(int, diff_str.split(',')))
                    diff_array = np.array(diff_list, dtype=np.int16).reshape((height, width))
                    img_array = np.zeros_like(diff_array)
                    img_array[0, 0] = diff_array[0, 0]
                    for i in range(1, height):
                        img_array[i, 0] = img_array[i-1, 0] + diff_array[i, 0]
                    for i in range(height):
                        for j in range(1, width):
                            img_array[i, j] = img_array[i, j-1] + diff_array[i, j]
                    channel_data.append(np.clip(img_array, 0, 255).astype(np.uint8))

                img = Image.merge('RGB', [Image.fromarray(arr, mode='L') for arr in channel_data])
        except FileNotFoundError:
            error_msg = f"Error: Compressed file '{input_path}' not found."
            print(error_msg)
            return None, error_msg
        except IOError as e:
            error_msg = f"Error reading file '{input_path}': {str(e)}"
            print(error_msg)
            return None, error_msg
        except Exception as e:
            error_msg = f"Error processing file '{input_path}': {str(e)}"
            print(error_msg)
            return None, error_msg

        try:
            img.save(output_path, image_format)
        except IOError as e:
            error_msg = f"Error saving image to '{output_path}': {str(e)}"
            print(error_msg)
            return None, error_msg

        stats = f"Color difference file '{os.path.basename(input_path)}' decompressed into '{output_file}'.\n"
        print(stats)
        return output_path, stats

    def compare_images(self, original_path, decompressed_path):
        """Compare original and decompressed images"""
        try:
            orig_img = Image.open(original_path).convert('RGB')
            dec_img = Image.open(decompressed_path).convert('RGB')
            
            if orig_img.size != dec_img.size:
                stats = "Images have different dimensions!\n"
                print(stats)
                return stats
                
            orig_array = np.array(orig_img)
            dec_array = np.array(dec_img)
            
            # Calculate differences
            diff_array = np.abs(orig_array - dec_array)
            total_pixels = orig_array.size // 3  # RGB has 3 channels
            pixel_differences = np.count_nonzero(diff_array)
            mean_diff = np.mean(diff_array)
            max_diff = np.max(diff_array)
            diff_percentage = (pixel_differences / total_pixels * 100)
            
            # Calculate PSNR (Peak Signal-to-Noise Ratio)
            mse = np.mean(diff_array ** 2)
            psnr = 20 * math.log10(255) - 10 * math.log10(mse) if mse > 0 else float('inf')
            
            stats = (f"Image Comparison:\n"
                    f"Original: {os.path.basename(original_path)}\n"
                    f"Decompressed: {os.path.basename(decompressed_path)}\n"
                    f"Dimensions: {orig_img.size}\n"
                    f"Pixel Differences: {pixel_differences} pixels\n"
                    f"Difference Percentage: {diff_percentage:.2f}%\n"
                    f"Mean Difference: {mean_diff:.2f}\n"
                    f"Max Difference: {max_diff}\n"
                    f"PSNR: {psnr:.2f} dB\n"
                    f"Files identical: {pixel_differences == 0}\n")
            print(stats)
            return stats
        except Exception as e:
            error_msg = f"Error comparing images: {str(e)}"
            print(error_msg)
            return error_msg

# GUI class
class CompressionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LZW Compression & Image Operations")
        self.root.geometry("900x700")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)

        self.text_tab = ttk.Frame(self.notebook)
        self.image_tab = ttk.Frame(self.notebook)
        self.editor_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.text_tab, text="Text Operations")
        self.notebook.add(self.image_tab, text="Image Compression")
        self.notebook.add(self.editor_tab, text="Image Editor")

        self.log_text = tk.Text(root, height=8)
        self.log_text.pack(fill='x')

        self.original_text_path = None
        self.original_image_path = None
        self.current_image_path = None  # For editor tab

        self.build_text_tab()
        self.build_image_tab()
        self.build_editor_tab()

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def build_text_tab(self):
        frame = ttk.Frame(self.text_tab, padding=10)
        frame.pack(fill='both', expand=True)
        ttk.Button(frame, text="Compress Text File", command=self.compress_text_gui).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(frame, text="Decompress Text File", command=self.decompress_text_gui).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Compare Text Files", command=self.compare_text_gui).grid(row=0, column=2, padx=5, pady=5)

    def compress_text_gui(self):
        file_path = filedialog.askopenfilename(title="Select Text File", filetypes=[("Text Files", "*.txt")])
        if file_path:
            self.original_text_path = file_path
            base = os.path.splitext(os.path.basename(file_path))[0]
            lzw = LZWCoding(base, "text", filepath=file_path)
            output_path, stats = lzw.compress_text_file()
            if output_path is None:
                self.log(stats)
            else:
                self.log(stats)

    def decompress_text_gui(self):
        file_path = filedialog.askopenfilename(title="Select Compressed File", filetypes=[("Binary Files", "*.bin")])
        if file_path:
            base = os.path.splitext(os.path.basename(file_path))[0]
            lzw = LZWCoding(base, "text", filepath=file_path)
            output_path, stats = lzw.decompress_text_file()
            if output_path is None:
                self.log(stats)
            else:
                self.log(stats)

    def compare_text_gui(self):
        if not self.original_text_path:
            self.log("Please compress a text file first to set original file path")
            return
            
        decompressed_path = filedialog.askopenfilename(
            title="Select Decompressed Text File", 
            filetypes=[("Text Files", "*.txt")]
        )
        if decompressed_path:
            lzw = LZWCoding("", "text")
            stats = lzw.compare_text_files(self.original_text_path, decompressed_path)
            self.log(stats)

    def build_image_tab(self):
        frame = ttk.Frame(self.image_tab, padding=10)
        frame.pack(fill='both', expand=True)
        
        # Add image display panel
        self.image_display = tk.Label(frame)
        self.image_display.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        buttons = [
            ("Compress Image (Raw)", self.compress_image_gui, 1, 0),
            ("Decompress Image (Raw)", self.decompress_image_gui, 1, 1),
            ("Compare Images (Raw)", self.compare_image_gui, 1, 2),
            ("Compress Gray Diff", self.compress_gray_gui, 2, 0),
            ("Decompress Gray Diff", self.decompress_gray_gui, 2, 1),
            ("Compare Gray Diff", self.compare_gray_gui, 2, 2),
            ("Compress Color Diff", self.compress_color_diff_gui, 3, 0),
            ("Decompress Color Diff", self.decompress_color_diff_gui, 3, 1),
            ("Compare Color Diff", self.compare_color_diff_gui, 3, 2)
        ]
        for text, command, row, col in buttons:
            ttk.Button(frame, text=text, command=command).grid(row=row, column=col, padx=5, pady=5)

    def display_image(self, path):
        """Display an image in the image compression tab."""
        try:
            img = Image.open(path)
            # Resize image to fit within a reasonable size (e.g., 300x300) if needed
            img.thumbnail((300, 300), Image.Resampling.LANCZOS)
            self.current_tk_image = ImageTk.PhotoImage(img)
            self.image_display.config(image=self.current_tk_image)
            self.image_display.image = self.current_tk_image  # Keep a reference to avoid garbage collection
        except Exception as e:
            self.log(f"Error displaying image: {str(e)}")

    def compress_image_gui(self):
        file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("BMP Files", "*.bmp")])
        if file_path:
            self.original_image_path = file_path
            base = os.path.splitext(os.path.basename(file_path))[0]
            lzw_img = LZWImageCoding(base, "image", filepath=file_path)
            output_path, stats = lzw_img.compress_image_file("bmp")
            if output_path is None:
                self.log(stats)
            else:
                self.log(stats)
                self.display_image(file_path)  # Display the original image after compression

    def decompress_image_gui(self):
        file_path = filedialog.askopenfilename(title="Select Compressed Image File", filetypes=[("Binary Files", "*.bin")])
        if file_path:
            base = os.path.splitext(os.path.basename(file_path))[0]
            lzw_img = LZWImageCoding(base, "image", filepath=file_path)
            output_path, stats = lzw_img.decompress_image_file("bmp")
            if output_path is None:
                self.log(stats)
            else:
                self.log(stats)
                self.display_image(output_path)  # Display the decompressed image

    def compare_image_gui(self):
        if not self.original_image_path:
            self.log("Please compress an image first to set original file path")
            return
        decompressed_path = filedialog.askopenfilename(
            title="Select Decompressed Image File", 
            filetypes=[("BMP Files", "*.bmp")]
        )
        if decompressed_path:
            lzw_img = LZWImageCoding("", "image")
            stats = lzw_img.compare_images(self.original_image_path, decompressed_path)
            self.log(stats)

    def compress_gray_gui(self):
        file_path = filedialog.askopenfilename(title="Select Grayscale Image File", filetypes=[("BMP Files", "*.bmp")])
        if file_path:
            self.original_image_path = file_path
            base = os.path.splitext(os.path.basename(file_path))[0]
            lzw_img = LZWImageCoding(base, "image", filepath=file_path)
            output_path, stats = lzw_img.compress_gray_image_diff("bmp")
            if output_path is None:
                self.log(stats)
            else:
                self.log(stats)
                self.display_image(file_path)  # Display the original grayscale image

    def decompress_gray_gui(self):
        file_path = filedialog.askopenfilename(title="Select GrayDiff Compressed File", filetypes=[("Binary Files", "*.bin")])
        if file_path:
            base = os.path.splitext(os.path.basename(file_path))[0]
            lzw_img = LZWImageCoding(base, "image", filepath=file_path)
            output_path, stats = lzw_img.decompress_gray_image_diff("bmp")
            if output_path is None:
                self.log(stats)
            else:
                self.log(stats)
                self.display_image(output_path)  # Display the decompressed grayscale image

    def compare_gray_gui(self):
        if not self.original_image_path:
            self.log("Please compress a grayscale image first to set original file path")
            return
        decompressed_path = filedialog.askopenfilename(
            title="Select Decompressed GrayDiff Image", 
            filetypes=[("BMP Files", "*.bmp")]
        )
        if decompressed_path:
            lzw_img = LZWImageCoding("", "image")
            stats = lzw_img.compare_images(self.original_image_path, decompressed_path)
            self.log(stats)

    def compress_color_diff_gui(self):
        file_path = filedialog.askopenfilename(title="Select Color Image File", filetypes=[("BMP Files", "*.bmp")])
        if file_path:
            self.original_image_path = file_path
            base = os.path.splitext(os.path.basename(file_path))[0]
            lzw_img = LZWImageCoding(base, "image", filepath=file_path)
            output_path, stats = lzw_img.compress_color_image_diff("bmp")
            if output_path is None:
                self.log(stats)
            else:
                self.log(stats)
                self.display_image(file_path)  # Display the original color image

    def decompress_color_diff_gui(self):
        file_path = filedialog.askopenfilename(title="Select ColorDiff Compressed File", filetypes=[("Binary Files", "*.bin")])
        if file_path:
            base = os.path.splitext(os.path.basename(file_path))[0]
            lzw_img = LZWImageCoding(base, "image", filepath=file_path)
            output_path, stats = lzw_img.decompress_color_image_diff("bmp")
            if output_path is None:
                self.log(stats)
            else:
                self.log(stats)
                self.display_image(output_path)  # Display the decompressed color image

    def compare_color_diff_gui(self):
        if not self.original_image_path:
            self.log("Please compress a color image first to set original file path")
            return
        decompressed_path = filedialog.askopenfilename(
            title="Select Decompressed ColorDiff Image", 
            filetypes=[("BMP Files", "*.bmp")]
        )
        if decompressed_path:
            lzw_img = LZWImageCoding("", "image")
            stats = lzw_img.compare_images(self.original_image_path, decompressed_path)
            self.log(stats)

    def build_editor_tab(self):
        frame = ttk.Frame(self.editor_tab, padding=10)
        frame.pack(fill='both', expand=True)
        self.editor_image_panel = tk.Label(frame)
        self.editor_image_panel.grid(row=0, column=0, columnspan=5, padx=10, pady=10)

        buttons = [
            ("Open Image", self.open_image_editor, 1, 0),
            ("Grayscale", self.display_in_grayscale, 1, 1),
            ("Red", lambda: self.display_color_channel('red'), 1, 2),
            ("Green", lambda: self.display_color_channel('green'), 1, 3),
            ("Blue", lambda: self.display_color_channel('blue'), 1, 4)
        ]
        for text, command, row, col in buttons:
            ttk.Button(frame, text=text, command=command).grid(row=row, column=col, padx=5, pady=5)

    def open_image_editor(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("BMP Files", "*.bmp")])
        if not file_path:
            messagebox.showinfo("Warning", "No image file selected!")
            return
        self.current_image_path = file_path
        self.display_image_editor(file_path)
        self.log(f"Opened image: {os.path.basename(file_path)}")

    def display_image_editor(self, path):
        try:
            img = Image.open(path)
            self.current_pil_image = img
            self.current_tk_image = ImageTk.PhotoImage(img)
            self.editor_image_panel.config(image=self.current_tk_image)
            self.editor_image_panel.image = self.current_tk_image
        except Exception as e:
            messagebox.showerror("Error", f"Unable to open image: {e}")

    def display_in_grayscale(self):
        if not self.current_image_path:
            messagebox.showinfo("Warning", "No image loaded!")
            return
        img = Image.open(self.current_image_path).convert('L')
        self.current_tk_image = ImageTk.PhotoImage(img)
        self.editor_image_panel.config(image=self.current_tk_image)
        self.editor_image_panel.image = self.current_tk_image
        self.log(f"Displayed grayscale image.")

    def display_color_channel(self, channel):
        if not self.current_image_path:
            messagebox.showinfo("Warning", "No image loaded!")
            return
        idx = {'red': 0, 'green': 1, 'blue': 2}[channel]
        img_rgb = Image.open(self.current_image_path)
        image_array = np.array(img_rgb)
        image_array[:, :, [i for i in range(3) if i != idx]] = 0
        new_img = Image.fromarray(np.uint8(image_array))
        self.current_tk_image = ImageTk.PhotoImage(new_img)
        self.editor_image_panel.config(image=self.current_tk_image)
        self.editor_image_panel.image = self.current_tk_image
        self.log(f"Displayed {channel} channel.")

if __name__ == "__main__":
    root = tk.Tk()
    app = CompressionApp(root)
    root.mainloop()
