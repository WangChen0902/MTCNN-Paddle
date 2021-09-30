from BBox_utils import getDataFromTxt, BBox

txt_path = '../data/widerface/train/label.txt'

re = getDataFromTxt(txt_path, with_landmark=True)
print(len(re))