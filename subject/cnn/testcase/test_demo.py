from unittest import TestCase
from testfile_demo import testthreeAPI

class TestThree(TestCase):

    def testthree_mut(self):
        #result = testfile.testoneAPI("E:/Program Files/PyCharm Community Edition 2020.1.1/projects/test1/image/test/300.jpg",0)
        #result = [[1,2],[1,2]]
        self.assertEqual(testthreeAPI("E:/Program Files/PyCharm Community Edition 2020.1.1/projects/CNN/train/","E:/Program Files/PyCharm Community Edition 2020.1.1/projects/CNN/test/",0), 1)