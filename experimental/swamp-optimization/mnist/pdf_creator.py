import os
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from PIL import Image as pilImage


def create_pdf(image_dir, pdf_filename, imagefile_sort_fn, images_per_row):
    all_images = []

    for parent, dirnames, filenames in os.walk(image_dir):
        for file_name in filenames:
            file_path = os.path.join(parent, file_name)
            if os.path.splitext(file_path)[1] == '.png':
                all_images.append(file_path)
    if len(all_images) > 0:
        _converted(pdf_filename, all_images, imagefile_sort_fn, images_per_row)


def _converted(pdf_file, all_images, imagefile_sort_fn, images_per_row):
    all_images.sort(key=imagefile_sort_fn)

    # Calculate scaling ratio
    origin_image_width, origin_image_height = pilImage.open(all_images[0]).size
    a4_width, a4_height = A4
    plot_image_width = a4_width / images_per_row
    ratio = plot_image_width / origin_image_width
    plot_image_height = origin_image_height * ratio

    elements = []
    pdf_doc = SimpleDocTemplate(
        pdf_file,
        pagesize=A4,
        rightMargin=0,
        leftMargin=0,
        topMargin=0,
        bottomMargin=0)
    data = []

    # Create images for table.
    img_idx = 0
    finish = False
    while not finish:
        row_data = []
        for i in range(images_per_row):
            if img_idx < len(all_images):
                image = all_images[img_idx]
                row_data.append(
                    Image(
                        image,
                        plot_image_width,
                        plot_image_height))
                img_idx += 1
            else:
                finish = True
                break
        if (len(row_data) > 0):
            data.append(row_data)

    table = Table(
        data,
        colWidths=plot_image_width,
        rowHeights=plot_image_height)
    table.setStyle(TableStyle([
        ('INNERGRID', (0, 0), (-1, -1), 0.25, colors.black),
        ('BOX', (0, 0), (-1, -1), 0.25, colors.black),
        ('BACKGROUND', (0, 0), (-1, -1), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
    ]))

    # Build PDF.
    elements.append(table)
    pdf_doc.build(elements)


def sort_image_file(filename):
    return filename.split('/')[-1].split('_')[3]


def main():
    create_pdf('jobs', 'metrics.pdf', sort_image_file, 2)


if __name__ == '__main__':
    main()
