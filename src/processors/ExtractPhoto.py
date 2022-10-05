"""
https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
"""

import cv2
import numpy as np
from src.utils.imgutils import four_point_transform, ImageUtils, MainOperations
from .interfaces.ImagePreprocessor import ImagePreprocessor
from .CropPage import normalize, validate_rect

class ExtractPhoto(ImagePreprocessor):
    def __init__(self, extract_ops, args):
        self.args = args
        self.morph_kernel   = tuple(extract_ops.get("morphKernel", [10, 10]))
        
        # todo: use these options in code
        self.file_suffix    = extract_ops.get("fileSuffix", "_extracted")
        self.area_origin = tuple(extract_ops.get("areaOrigin", [0, 0]))
        self.area_dimensions = tuple(extract_ops.get("areaDimensions", [200, 200]))
        self.min_photo_area = extract_ops.get("minPhotoArea", 2000)
        
        # todo: take more arguments from extract_ops if needed (e.g. canny parameters)


    def find_page(self, image):
        image = normalize(image)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel)

        # Close the small holes, i.e. Complete the edges on canny image
        closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        MainOperations.show( "closed", closed, 1)

        # TODO: Parametrize this from template config
        edge = cv2.Canny(closed, 185, 55)
        MainOperations.show( "canny", edge, 1)

        # findContours returns outer boundaries in CW and inner boundaries in ACW
        # order.
        cnts = ImageUtils.grab_contours(
            cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        )
        cnts = [cv2.convexHull(c) for c in cnts]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        sheet = []
        for c in cnts:
            if cv2.contourArea(c) < self.min_photo_area:
                continue
            peri = cv2.arcLength(c, True)

            approx = cv2.approxPolyDP(c, epsilon=0.025 * peri, closed=True)

            # check its rectangle-ness:
            if validate_rect(approx):
                sheet = np.reshape(approx, (4, -1))
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
                cv2.drawContours(edge, [approx], -1, (255, 255, 255), 10)
                break

        return sheet

    # todo: make sure we get colored image as one more argument here. So as to save colored output.
    def apply_filter(self, original_image, _args):

        image = normalize(cv2.GaussianBlur(original_image, (3, 3), 0))
        
        # todo: consume area_origins and area_dimensions to crop the image first

        # Resize should be done with another preprocessor is needed
        sheet = self.find_page(image)

        if sheet == []:
            print(
                "\tError: Photo boundary not found!"
            )
            return None

        print("Found photo corners: \t", sheet.tolist())

        # Warp layer 1
        image = four_point_transform(image, sheet)
        
        print(sheet)
        # todo: save image into correct output path
        MainOperations.show( "image", image, 1)

        # Return original image
        return original_image
