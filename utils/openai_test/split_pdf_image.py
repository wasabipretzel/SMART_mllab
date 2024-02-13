"""
    given instruction manual pdf, create folder and store images refer to each page.
"""
import os
from PIL import Image
from pdf2image import convert_from_path


# PDF 파일을 이미지로 변환
pdf_file = '/data/IKEA/dataset/Furniture/Beds/90371954/manual/gulliver-cot-white__AA-1989121-3.pdf'
images = convert_from_path(pdf_file)

# Define the maximum dimension (512x512)
max_dimension = 512

# create save folder
save_folder = '/openai_test/test/'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

# resize image to desired size maintaining its ratio

for i, image in enumerate(images):
    page_num = i+1
    original_width, original_height = image.size

    # Calculate the new dimensions while maintaining aspect ratio
    if original_width > original_height:
        new_width = max_dimension
        new_height = int(original_height * (max_dimension / original_width))
    else:
        new_height = max_dimension
        new_width = int(original_width * (max_dimension / original_height))

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    print(f"resized {page_num} size : {resized_image.size}")
    # Save the resized image
    resized_image.save(os.path.join(save_folder,f"{page_num}.jpg"))  # Replace "resized_image.jpg" with your desired output file path
