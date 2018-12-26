import os
from reportlab.platypus import SimpleDocTemplate, Image
from reportlab.lib.pagesizes import A4, landscape
from PIL import Image as pilImage


def create_pdf(image_dir, pdf_filename, imagefile_sort_fn):
    all_images = []

    for parent, dirnames, filenames in os.walk(image_dir):
        for file_name in filenames:
            file_path = os.path.join(parent, file_name)
            if os.path.splitext(file_path)[1] == '.png':
                all_images.append(file_path)
    if len(all_images) > 0:
        _converted(pdf_filename, all_images, imagefile_sort_fn)

def _converted(pdf_file, all_images, imagefile_sort_fn):
    all_images.sort(key=imagefile_sort_fn)
    imagesData = []
    ratio = 0.5

    pdf_doc = SimpleDocTemplate(
        pdf_file,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18)

    for image in all_images:
        img_w, img_h = pilImage.open(image).size
        data = Image(image, img_w * ratio, img_h * ratio)
        imagesData.append(data)
    pdf_doc.build(imagesData)

def sort_image_file(filename):
    return filename.split('/')[-1].split('_')[3]   

def main():
    create_pdf('jobs', 'metrics.pdf', sort_image_file)

if __name__ == '__main__':
    main()
