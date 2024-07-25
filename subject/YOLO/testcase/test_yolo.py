from unittest import TestCase
from testfile_yolo import testAPI

class TestFour(TestCase):

    def testfour_mut(self):
        #result = testfile.testoneAPI("E:/Program Files/PyCharm Community Edition 2020.1.1/projects/test1/image/test/300.jpg",0)
        #result = [[1,2],[1,2]]
        self.assertEqual(testAPI("D:/Amy/program/ImageProcessingSoftware/new_epm/TestSubject/YOLO/show_image/test/000000000139.jpg"), 0)