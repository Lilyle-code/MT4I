#python E:\anaconda3\envs\python\Scripts\mut.py --target testfile_expand1 --unit-test test_expand1 -m --coverage > mut1.txt
from unittest import TestCase
#import expand1
from testfile_expand1 import testoneAPI
#from testfile_sort import testtwoAPI

class TestOne(TestCase):

    def testone_mut(self):
        #result = testfile.testoneAPI("E:/Program Files/PyCharm Community Edition 2020.1.1/projects/test1/image/test/300.jpg",0)
        #result = [[1,2],[1,2]]
        self.assertEqual(testoneAPI("E:/Program Files/PyCharm Community Edition 2020.1.1/projects/test1/image/test/300.jpg",0), 1)
        #self.assertEqual(testtwoAPI("E:/Program Files/PyCharm Community Edition 2020.1.1/projects/KnnSort/Corel-1000/train","E:/Program Files/PyCharm Community Edition 2020.1.1/projects/KnnSort/Corel-1000/test",0), 1)
