from unittest import TestCase
from testfile_sort import testtwoAPI

class TestTwo(TestCase):

    def testtwo_mut(self):
        #result = testfile.testoneAPI("E:/Program Files/PyCharm Community Edition 2020.1.1/projects/test1/image/test/300.jpg",0)
        #result = [[1,2],[1,2]]
        self.assertEqual(testtwoAPI("E:/Program Files/PyCharm Community Edition 2020.1.1/projects/KnnSort/Corel-1000/test","E:/Program Files/PyCharm Community Edition 2020.1.1/projects/KnnSort/Corel-1000/train",0), 1)