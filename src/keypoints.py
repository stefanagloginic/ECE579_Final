from csv import DictReader
from PIL import ImageColor

def get_keypoint_definitions(keypoint_def_file_path):
    keypoint_def_file = open(keypoint_def_file_path)
    keypoint_def = DictReader(keypoint_def_file)

    keypoint_colors = []
    keypoint_labels = []
    for row in keypoint_def:
        keypoint_color = ImageColor.getcolor(f"#{row['Hex colour']}", "RGB")
        keypoint_colors.append(keypoint_color)
        keypoint_labels.append(row['Name'])

    return {
        'colors': keypoint_colors,
        'labels': keypoint_labels
    }


def is_valid_keypoint(keypoint):
    ## According to the document https://github.com/benjiebob/StanfordExtra/blob/master/demo.ipynb
    ## non labeled keypoints are stored as [0., 0., 0.]
    ## Some non visible keypoints might be useful
    x, y, visibility = keypoint
    return not (x == 0 and y == 0 and visibility == 0)

